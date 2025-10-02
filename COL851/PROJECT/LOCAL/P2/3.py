#!/usr/bin/env python3
"""
chronos_experiments_exporter_granular_v2.py

This is the updated version that writes per-predict JSONL and *updates the
Prometheus gauges immediately after each predict*. The gauges reflect the
latest predict for the label tuple (city, model, context_days, horizon_hours),
so Grafana panels using those gauges will appear live during a sweep.

See the canvas file titled 'Chronos Experiments Exporter Granular' for the
original version. This file focuses on live gauge updates per predict.
"""

import os
import time
import csv
import json
import threading
import subprocess
from pathlib import Path
from statistics import mean, median
import math

import numpy as np
import torch
import pandas as pd
import psutil
from prometheus_client import start_http_server, Gauge

# Chronos inference
from chronos import BaseChronosPipeline

# ---- CONFIG ----
CITY_FILES = {"gurgaon": "gurgaon.csv", "patna": "patna.csv"}
MODEL_NAME = "amazon/chronos-bolt-tiny"
DEVICE = "cpu"
OUT_DIR = "chronos_results"
os.makedirs(OUT_DIR, exist_ok=True)
PER_PREDICT_JSONL = os.path.join(OUT_DIR, "per_predict_series.jsonl")
AGG_CSV = os.path.join(OUT_DIR, "chronos_experiment_results_agg.csv")
EXPORTER_PORT = 8001

# sampling / perf
SAMPLE_INTERVAL = 0.05   # seconds between psutil samples inside each predict
USE_PERF = True
USE_RAPL = True

# evaluation sweeps
CONTEXT_DAYS_LIST = [2, 4, 8, 10, 14]
HORIZONS_HOURS_LIST = [4, 8, 12, 24, 48]
STEP_SIZE = None  # sliding windows step
RESAMPLE_HOURLY = False

# caching pipeline
_pipeline_cache = {}

# Prometheus gauges (summary-level & per-predict current values)
LATENCY_AVG_G = Gauge("chronos_run_latency_s_avg", "Avg predict latency (s)", ["city","model","context_days","horizon_hours"])
LATENCY_P50_G = Gauge("chronos_run_latency_s_p50", "P50 predict latency (s)", ["city","model","context_days","horizon_hours"])
THROUGHPUT_AVG_G = Gauge("chronos_run_throughput_items_per_s_avg", "Avg throughput (pred_len / latency)", ["city","model","context_days","horizon_hours"])
CACHE_MISSES_SUM_G = Gauge("chronos_run_cache_misses_sum", "Sum of perf cache-misses across predicts", ["city","model","context_days","horizon_hours"])
CPU_SERIES_AVAILABLE_G = Gauge("chronos_run_cpu_series_available", "Flag if per-predict cpu series were recorded (1=yes)", ["city","model","context_days","horizon_hours"])

# Per-predict current gauges (updated after each predict for the running labels)
LATENCY_CUR_G = Gauge("chronos_predict_latency_s", "Most-recent predict latency (s)", ["city","model","context_days","horizon_hours"])
THROUGHPUT_CUR_G = Gauge("chronos_predict_throughput_items_per_s", "Most-recent predict throughput (items/s)", ["city","model","context_days","horizon_hours"])
PERF_MISSES_CUR_G = Gauge("chronos_predict_perf_cache_misses", "Most-recent predict cache-misses", ["city","model","context_days","horizon_hours"])
ENERGY_CUR_G = Gauge("chronos_predict_energy_j", "Most-recent predict energy (J)", ["city","model","context_days","horizon_hours"])
RMSE_CUR_G = Gauge("chronos_predict_rmse", "Most-recent predict RMSE", ["city","model","context_days","horizon_hours"])

# --- utilities ---

def load_pipeline(model_name=MODEL_NAME, device=DEVICE):
    key = (model_name, device)
    if key in _pipeline_cache:
        return _pipeline_cache[key]
    print(f"Loading Chronos pipeline {model_name} on device={device} ...")
    pipeline = BaseChronosPipeline.from_pretrained(model_name, device_map=device)
    _pipeline_cache[key] = pipeline
    return pipeline


def read_city_file(path, resample_hourly=RESAMPLE_HOURLY):
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    dt_candidates = ["From Date"]
    pm_candidates = ["calibPM"]
    dt_col = next((c for c in dt_candidates if c in df.columns), None)
    if dt_col is None:
        dt_col = df.columns[0]
    pm_col = next((c for c in pm_candidates if c in df.columns), None)
    if pm_col is None:
        pm_col = df.columns[-1]
    ts = pd.to_datetime(df[dt_col], errors="coerce", infer_datetime_format=True)
    valid_mask = ~ts.isna()
    if valid_mask.sum() < len(df):
        df = df.loc[valid_mask].copy()
        ts = ts.loc[valid_mask]
    vals = pd.to_numeric(df[pm_col], errors="coerce")
    vals = vals.ffill().bfill()
    if resample_hourly:
        s = pd.Series(vals.values, index=ts)
        s = s.sort_index()
        s = s.resample("H").mean().interpolate(limit_direction="both")
        return s.index.to_numpy(), s.values
    return ts.to_numpy(), vals.to_numpy()


def read_rapl_joules():
    if not USE_RAPL:
        return None
    import glob
    total_uj = 0
    for p in glob.glob("/sys/class/powercap/intel-rapl:*/*/energy_uj"):
        try:
            with open(p, "r") as f:
                total_uj += int(f.read().strip())
        except Exception:
            pass
    return total_uj / 1e6 if total_uj else None


# ---- instrumentation helpers ----

def _sample_psutil(stop_event, interval, samples, start_time):
    proc = psutil.Process(os.getpid())
    # Prime cpu_percent counters
    psutil.cpu_percent(interval=None)
    proc.cpu_percent(interval=None)
    while not stop_event.is_set():
        t = time.time() - start_time
        cpu_total = psutil.cpu_percent(interval=None)
        proc_cpu = proc.cpu_percent(interval=None) / (psutil.cpu_count(logical=True) or 1)
        mem_used = psutil.virtual_memory().used
        rss = proc.memory_info().rss
        samples.append({"t": round(t, 6), "cpu_total": cpu_total, "proc_cpu": proc_cpu, "mem_used": mem_used, "rss": rss})
        slept = 0.0
        while slept < interval and not stop_event.is_set():
            time.sleep(min(0.01, interval - slept))
            slept += min(0.01, interval - slept)


def _start_perf_for_pid(pid):
    try:
        proc = subprocess.Popen(['perf','stat','-e','cache-references,cache-misses','-p', str(pid), '--','sleep','3600'],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return proc
    except Exception:
        return None


def _stop_and_parse_perf(perf_proc, timeout=5):
    if perf_proc is None:
        return None
    try:
        perf_proc.terminate()
        stderr, stdout = perf_proc.communicate(timeout=timeout)
        import re
        m = re.search(r'([0-9][0-9,]*)\s+cache-misses', stderr or "")
        if m:
            return float(m.group(1).replace(',', ''))
    except Exception:
        pass
    return None


def point_forecast_from_pipeline_output(forecast):
    if isinstance(forecast, torch.Tensor):
        forecast = forecast.cpu().numpy()
    arr = np.array(forecast)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        q_count = arr.shape[0]
        if q_count % 2 == 1:
            return arr[q_count // 2]
        else:
            return arr.mean(axis=0)
    if arr.ndim == 3:
        arr0 = arr[0]
        if arr0.ndim == 2:
            q_count = arr0.shape[0]
            if q_count % 2 == 1:
                return arr0[q_count // 2]
            else:
                return arr0.mean(axis=0)
    return arr.mean(axis=tuple(range(arr.ndim - 1)))


# Instrumented predict
def instrumented_predict(pipeline, context_tensor, prediction_length, sample_interval=SAMPLE_INTERVAL, use_perf=USE_PERF):
    stop_event = threading.Event()
    samples = []
    start_time = time.time()
    sampler = threading.Thread(target=_sample_psutil, args=(stop_event, sample_interval, samples, start_time), daemon=True)

    perf_proc = _start_perf_for_pid(os.getpid()) if use_perf else None

    energy_before = read_rapl_joules()

    sampler.start()
    try:
        t0 = time.time()
        with torch.no_grad():
            forecast = pipeline.predict(context=context_tensor, prediction_length=prediction_length)
        t1 = time.time()
    finally:
        stop_event.set()
        sampler.join(timeout=2)
        perf_misses = _stop_and_parse_perf(perf_proc, timeout=2) if perf_proc else None
        energy_after = read_rapl_joules()

    latency = t1 - t0
    pred = point_forecast_from_pipeline_output(forecast)
    pred = np.asarray(pred, dtype=float)
    if pred.shape[0] != prediction_length:
        pred = np.resize(pred, prediction_length)

    energy_delta = None
    if energy_before is not None and energy_after is not None:
        energy_delta = float(energy_after - energy_before)

    throughput = None
    try:
        throughput = float(prediction_length) / max(1e-9, latency)
    except Exception:
        throughput = None

    metrics = {
        "latency_s": latency,
        "throughput_items_per_s": throughput,
        "energy_j": energy_delta,
        "perf_cache_misses": perf_misses,
        "psutil_samples": samples,
        "n_psamples": len(samples),
        "predict_time": start_time,
    }
    return pred, metrics


# Streaming evaluator: writes per-predict JSONL and updates per-predict gauges live
def evaluate_series_zero_shot_instrumented(values, context_len_hours, pred_len_hours, pipeline, step=None,
                                            city=None, model_name=None, context_days=None, horizon_hours=None):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < context_len_hours + pred_len_hours:
        return []
    if step is None:
        step = pred_len_hours
    results = []
    starts = list(range(context_len_hours, n - pred_len_hours + 1, step))

    prom_labels = {"city": city or "", "model": (model_name or "").replace("/","_"),
                   "context_days": str(context_days or context_len_hours//24), "horizon_hours": str(horizon_hours or pred_len_hours)}

    for start in starts:
        hist = values[start - context_len_hours : start]
        true_fut = values[start : start + pred_len_hours]
        if len(hist) != context_len_hours or len(true_fut) != pred_len_hours:
            continue
        ctx_tensor = torch.tensor(hist, dtype=torch.float32)
        try:
            pred, metrics = instrumented_predict(pipeline, ctx_tensor, pred_len_hours)
        except Exception as e:
            print("predict failed for window start", start, "err:", e)
            continue
        if np.isnan(pred).any() or np.isnan(true_fut).any():
            continue
        mse = ((true_fut - pred) ** 2).mean()
        rmse = float(math.sqrt(mse))
        entry = {
            "start_idx": int(start),
            "rmse": rmse,
            "pred": pred.tolist(),
            "true": true_fut.tolist(),
            **metrics,
        }

        # immediate JSONL emission
        meta = {"city": city, "model": model_name, "context_days": context_days, "horizon_hours": horizon_hours}
        out = {**meta, **entry}
        try:
            with open(PER_PREDICT_JSONL, "a") as jf:
                jf.write(json.dumps(out) + "\n")
        except Exception as e:
            print("warning: failed to write per-predict jsonl:", e)

        # update live gauges
        try:
            LATENCY_CUR_G.labels(**prom_labels).set(metrics.get("latency_s") or 0.0)
            THROUGHPUT_CUR_G.labels(**prom_labels).set(metrics.get("throughput_items_per_s") or 0.0)
            PERF_MISSES_CUR_G.labels(**prom_labels).set(float(metrics.get("perf_cache_misses") or 0.0))
            ENERGY_CUR_G.labels(**prom_labels).set(metrics.get("energy_j") or 0.0)
            RMSE_CUR_G.labels(**prom_labels).set(rmse)
            CPU_SERIES_AVAILABLE_G.labels(**prom_labels).set(1 if metrics.get("n_psamples",0) > 0 else 0)
        except Exception as e:
            print("warning: failed to set per-predict gauges:", e)

        results.append(entry)
    return results


# Aggregation helpers

def aggregate_predict_series(predict_entries):
    latencies = [e["latency_s"] for e in predict_entries if e.get("latency_s") is not None]
    throughputs = [e["throughput_items_per_s"] for e in predict_entries if e.get("throughput_items_per_s") is not None]
    perf_misses = [e.get("perf_cache_misses") for e in predict_entries if e.get("perf_cache_misses") is not None]
    energy = [e.get("energy_j") for e in predict_entries if e.get("energy_j") is not None]
    n_series_flag = 1 if any(e.get("n_psamples", 0) > 0 for e in predict_entries) else 0
    return {
        "n_windows": len(predict_entries),
        "latency_avg": float(mean(latencies)) if latencies else float("nan"),
        "latency_p50": float(median(latencies)) if latencies else float("nan"),
        "throughput_avg": float(mean(throughputs)) if throughputs else float("nan"),
        "perf_misses_sum": float(sum(perf_misses)) if perf_misses else float("nan"),
        "energy_j_sum": float(sum(energy)) if energy else float("nan"),
        "cpu_series_recorded": n_series_flag,
    }


# run one experiment config and return aggregated metrics

def run_one_experiment_granular(city, vals, model_name, context_days, horizon_hours, pipeline, repeat_windows_step=None):
    context_hours = int(context_days) * 24
    pred_len = int(horizon_hours)
    entries = evaluate_series_zero_shot_instrumented(vals, context_hours, pred_len, pipeline, step=repeat_windows_step,
                                                     city=city, model_name=model_name, context_days=context_days, horizon_hours=horizon_hours)

    agg = aggregate_predict_series(entries)
    rmses = [e["rmse"] for e in entries]
    avg_rmse = float(mean(rmses)) if rmses else float("nan")
    std_rmse = float(np.std(rmses)) if rmses else float("nan")
    agg_row = {
        "city": city,
        "model": model_name,
        "context_days": context_days,
        "horizon_hours": horizon_hours,
        "avg_rmse": avg_rmse,
        "std_rmse": std_rmse,
        **agg,
    }
    return agg_row


# orchestration main

def main():
    start_http_server(EXPORTER_PORT)
    print("Exporter on port", EXPORTER_PORT)

    pipeline = load_pipeline(MODEL_NAME, DEVICE)

    write_header = not os.path.exists(AGG_CSV)
    with open(AGG_CSV, "a", newline="") as cf:
        writer = csv.writer(cf)
        if write_header:
            writer.writerow([
                "city","model","context_days","horizon_hours","n_windows","avg_rmse","std_rmse",
                "latency_avg_s","latency_p50_s","throughput_avg_items_s","perf_misses_sum","energy_j_sum","cpu_series_recorded","timestamp"])

    if not os.path.exists(PER_PREDICT_JSONL):
        open(PER_PREDICT_JSONL, "w").close()

    all_city_series = {}
    for city, fname in CITY_FILES.items():
        ts, vals = read_city_file(fname, resample_hourly=RESAMPLE_HOURLY)
        all_city_series[city] = (ts, vals)

    for city, (ts, vals) in all_city_series.items():
        for model in [MODEL_NAME]:
            for ctx_days in CONTEXT_DAYS_LIST:
                res = run_one_experiment_granular(city, vals, model, ctx_days, 24, pipeline, repeat_windows_step=STEP_SIZE)
                labels = {"city": city, "model": model.replace("/", "_"), "context_days": str(ctx_days), "horizon_hours": "24"}

                LATENCY_AVG_G.labels(**labels).set(res.get("latency_avg", 0.0) if res.get("latency_avg") == res.get("latency_avg") else 0.0)
                LATENCY_P50_G.labels(**labels).set(res.get("latency_p50", 0.0))
                THROUGHPUT_AVG_G.labels(**labels).set(res.get("throughput_avg", 0.0))
                CACHE_MISSES_SUM_G.labels(**labels).set(res.get("perf_misses_sum", 0.0) if res.get("perf_misses_sum") == res.get("perf_misses_sum") else 0.0)
                CPU_SERIES_AVAILABLE_G.labels(**labels).set(res.get("cpu_series_recorded", 0))

                with open(AGG_CSV, "a", newline="") as cf:
                    writer = csv.writer(cf)
                    writer.writerow([
                        res.get("city"), res.get("model"), res.get("context_days"), res.get("horizon_hours"), res.get("n_windows"),
                        res.get("avg_rmse"), res.get("std_rmse"), res.get("latency_avg"), res.get("latency_p50"), res.get("throughput_avg"),
                        res.get("perf_misses_sum"), res.get("energy_j_sum"), res.get("cpu_series_recorded"), time.time()])
                    cf.flush()
                print("wrote row for", labels)

    for city,(ts,vals) in all_city_series.items():
        for model in [MODEL_NAME]:
            for horizon in HORIZONS_HOURS_LIST:
                res = run_one_experiment_granular(city, vals, model, 10, horizon, pipeline)
                labels = {"city": city, "model": model.replace("/", "_"), "context_days": "10", "horizon_hours": str(horizon)}
                LATENCY_AVG_G.labels(**labels).set(res.get("latency_avg", 0.0) if res.get("latency_avg") == res.get("latency_avg") else 0.0)
                LATENCY_P50_G.labels(**labels).set(res.get("latency_p50", 0.0))
                THROUGHPUT_AVG_G.labels(**labels).set(res.get("throughput_avg", 0.0))
                CACHE_MISSES_SUM_G.labels(**labels).set(res.get("perf_misses_sum", 0.0) if res.get("perf_misses_sum") == res.get("perf_misses_sum") else 0.0)
                CPU_SERIES_AVAILABLE_G.labels(**labels).set(res.get("cpu_series_recorded", 0))
                with open(AGG_CSV, "a", newline="") as cf:
                    writer = csv.writer(cf)
                    writer.writerow([
                        res.get("city"), res.get("model"), res.get("context_days"), res.get("horizon_hours"), res.get("n_windows"),
                        res.get("avg_rmse"), res.get("std_rmse"), res.get("latency_avg"), res.get("latency_p50"), res.get("throughput_avg"),
                        res.get("perf_misses_sum"), res.get("energy_j_sum"), res.get("cpu_series_recorded"), time.time()])
                    cf.flush()
                print("wrote row for", labels)

    print("All experiments finished. Per-predict JSONL at", PER_PREDICT_JSONL)
    print("Aggregated CSV at", AGG_CSV)


if __name__ == "__main__":
    main()
