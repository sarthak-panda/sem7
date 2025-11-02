

#!/usr/bin/env python3
"""
chronos_zero_shot_eval.py

Usage:
  Put gurgaon.csv and patna.csv in the same directory.
  Run: python chronos_zero_shot_eval.py

Notes:
 - The reader auto-detects timestamp and PM columns (handles headers like "From Date" and "calibPM").
 - Missing PM values are filled with forward-fill then back-fill using .ffill().bfill().
 - If your timestamps are irregular and you'd like strict hourly resampling/interpolation,
   set RESAMPLE_HOURLY = True below.
"""
import os
import math
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Chronos inference
from chronos import BaseChronosPipeline

# ---- CONFIG ----
CITY_FILES = {"gurgaon": "gurgaon.csv", "patna": "patna.csv"}
# Model choices: chronos-bolt-tiny / mini / small / base; base is largest listed (205M).
MODEL_NAME = "amazon/chronos-bolt-tiny"   # change if too big for your laptop
DEVICE = "cpu"                            # run on CPU
# evaluation parameters
CONTEXT_DAYS_LIST = [2, 4, 8, 10, 14]    # for plot 1 (days)
HORIZONS_HOURS_LIST = [4, 8, 12, 24, 48] # for plot 2 (hours)
STEP_SIZE = None  # None -> defaults to prediction_length (non-overlapping windows); can set to 1 for dense sliding windows
# save outputs
OUT_DIR = "chronos_results"
os.makedirs(OUT_DIR, exist_ok=True)

# If True, the reader will resample the series to strict hourly frequency and interpolate missing values.
RESAMPLE_HOURLY = False

# ---- Utilities ----
_pipeline_cache = {}

def load_pipeline(model_name=MODEL_NAME, device=DEVICE):
    """Load & cache a chronos pipeline for inference."""
    key = (model_name, device)
    if key in _pipeline_cache:
        return _pipeline_cache[key]
    print(f"Loading Chronos pipeline {model_name} on device={device} ... (this can take a bit)")
    pipeline = BaseChronosPipeline.from_pretrained(model_name, device_map=device)
    _pipeline_cache[key] = pipeline
    return pipeline

def read_city_file(path, resample_hourly=RESAMPLE_HOURLY):
    """
    Read CSV and return (timestamps_np, values_np).
    - Auto-detects datetime and pm columns when header exists.
    - Falls back to first column as datetime and last column as PM if names not found.
    - Parses timestamps and coerces PM to numeric.
    - Fills missing values using forward-fill then back-fill using .ffill().bfill().
    - Optionally resamples to hourly frequency and interpolates.
    """
    df = pd.read_csv(path)
    # normalize column names to strings
    df.columns = [str(c).strip() for c in df.columns]

    # possible column name candidates for datetime and pm
    dt_candidates = ["From Date"]
    pm_candidates = ["calibPM"]

    # detect datetime column
    dt_col = None
    for c in dt_candidates:
        if c in df.columns:
            dt_col = c
            break
    if dt_col is None:
        # fallback to first column
        dt_col = df.columns[0]

    # detect pm column
    pm_col = None
    for c in pm_candidates:
        if c in df.columns:
            pm_col = c
            break
    if pm_col is None:
        # fallback to last column
        pm_col = df.columns[-1]

    # parse timestamps (coerce invalid -> NaT)
    ts = pd.to_datetime(df[dt_col], errors="coerce", infer_datetime_format=True)

    # drop rows with invalid timestamps
    valid_mask = ~ts.isna()
    if valid_mask.sum() < len(df):
        # keep only valid timestamp rows
        df = df.loc[valid_mask].copy()
        ts = ts.loc[valid_mask]

    # numeric PM values, coerce errors to NaN
    vals = pd.to_numeric(df[pm_col], errors="coerce")

    # fill missing numeric PM values (forward then backward)
    vals = vals.ffill().bfill()

    # If resample_hourly requested, resample to strict hourly index and interpolate missing
    if resample_hourly:
        s = pd.Series(vals.values, index=ts)
        # sort index first
        s = s.sort_index()
        # resample to hourly, take mean inside bucket, then interpolate small gaps
        s = s.resample("H").mean().interpolate(limit_direction="both")
        return s.index.to_numpy(), s.values

    # return arrays aligned with ts order
    # ensure both are numpy arrays with same length
    return ts.to_numpy(), vals.to_numpy()

def point_forecast_from_pipeline_output(forecast):
    """
    Convert pipeline output to 1-D point forecast (length = prediction_length).
    Strategy: if multiple quantiles are returned, take the median quantile if present,
    otherwise take mean across quantiles.
    """
    # convert to numpy if torch tensor
    if isinstance(forecast, torch.Tensor):
        forecast = forecast.cpu().numpy()
    arr = np.array(forecast)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        # (num_quantiles, pred_len)
        q_count = arr.shape[0]
        if q_count % 2 == 1:
            return arr[q_count // 2]
        else:
            return arr.mean(axis=0)
    if arr.ndim == 3:
        # (batch, num_quantiles, pred_len) -> take first batch
        arr0 = arr[0]
        if arr0.ndim == 2:
            q_count = arr0.shape[0]
            if q_count % 2 == 1:
                return arr0[q_count // 2]
            else:
                return arr0.mean(axis=0)
    # fallback: mean across all but last dim
    return arr.mean(axis=tuple(range(arr.ndim - 1)))

def evaluate_series_zero_shot(values, context_len_hours, pred_len_hours, pipeline, step=None):
    """
    Rolling-origin evaluation.
    values: 1D numpy array of hourly observations
    context_len_hours, pred_len_hours: ints
    step: sliding step in hours (if None, use pred_len_hours)
    Returns: list of RMSE values (one per evaluation window)
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < context_len_hours + pred_len_hours:
        return []  # not enough length
    if step is None:
        step = pred_len_hours

    rmses = []
    starts = list(range(context_len_hours, n - pred_len_hours + 1, step))
    for start in starts:
        hist = values[start - context_len_hours : start]
        true_fut = values[start : start + pred_len_hours]

        # make sure lengths are correct
        if len(hist) != context_len_hours or len(true_fut) != pred_len_hours:
            # skip malformed window
            continue

        ctx_tensor = torch.tensor(hist, dtype=torch.float32)
        with torch.no_grad():
            forecast = pipeline.predict(context=ctx_tensor, prediction_length=pred_len_hours)
        pred = point_forecast_from_pipeline_output(forecast)

        # ensure pred is a 1D numpy array of correct length
        pred = np.asarray(pred, dtype=float)
        if pred.shape[0] != pred_len_hours:
            pred = np.resize(pred, pred_len_hours)

        # If either contains NaNs, skip this window
        if np.isnan(pred).any() or np.isnan(true_fut).any():
            continue

        # compute RMSE without using 'squared' kwarg (works on older sklearn)
        mse = mean_squared_error(true_fut, pred)  # returns MSE
        rmse = float(np.sqrt(mse))
        rmses.append(rmse)

    return rmses

def avg_rmse_for_contexts_and_horizon(values, context_days_list, horizon_hours, pipeline, step=None):
    results = {}
    for days in context_days_list:
        context_hours = int(days * 24)
        rmses = evaluate_series_zero_shot(values, context_hours, horizon_hours, pipeline, step=step)
        results[days] = float(np.mean(rmses)) if len(rmses) > 0 else float("nan")
    return results

def avg_rmse_for_horizons(values, context_days, horizons_hours_list, pipeline, step=None):
    context_hours = int(context_days * 24)
    results = {}
    for h in horizons_hours_list:
        rmses = evaluate_series_zero_shot(values, context_hours, h, pipeline, step=step)
        results[h] = float(np.mean(rmses)) if len(rmses) > 0 else float("nan")
    return results








#!/usr/bin/env python3
"""
chronos_experiments_exporter.py
- Runs the RMSE sweeps (context/horizon/model) using Chronos pipeline functions you already have.
- Samples psutil / RAPL / optional `perf` while each run executes and exposes aggregated
  per-run metrics to Prometheus on port 8001.
- Writes a CSV of per-run aggregates for plotting/reporting.
"""
import os
import time
import csv
import threading
import subprocess
from pathlib import Path
from statistics import mean, pstdev
import numpy as np
import torch
import pandas as pd
import psutil

# plotting & metrics
import matplotlib.pyplot as plt
from prometheus_client import start_http_server, Gauge

# import your Chronos pipeline-loading and evaluation helpers
# (you can copy functions load_pipeline, read_city_file, evaluate_series_zero_shot, point_forecast_from_pipeline_output)
# For brevity assume they are defined in this file (paste yours), or import them.

# --------------- CONFIG ---------------
OUT_DIR = "chronos_results"
os.makedirs(OUT_DIR, exist_ok=True)
EXPORTER_PORT = 8001
SAMPLE_INTERVAL = 0.5   # seconds between psutil samples during a run
PERF_SAMPLE_INTERVAL = 0.5   # seconds for a one-shot perf stat (optional)
USE_PERF = True   # set False if perf isn't available / elevated privileges required
# --------------------------------------

# Define Prometheus gauges (labels are minimal to avoid cardinality explosion)
RMSE_G = Gauge("chronos_run_rmse", "Average RMSE for the run", ["city","model","context_days","horizon_hours"])
CPU_AVG_G = Gauge("chronos_run_cpu_percent_avg", "Avg system CPU (%) during run", ["city","model","context_days","horizon_hours"])
RSS_AVG_G = Gauge("chronos_run_process_rss_bytes_avg", "Avg process RSS bytes during run", ["city","model","context_days","horizon_hours"])
ENERGY_DELTA_G = Gauge("chronos_run_energy_joules", "Energy delta (joules) during run", ["city","model","context_days","horizon_hours"])
CACHE_MISSES_G = Gauge("chronos_run_cache_misses", "Cache misses measured during run", ["city","model","context_days","horizon_hours"])

# helper: read RAPL
def read_rapl_joules():
    import glob
    total_uj = 0
    for p in glob.glob("/sys/class/powercap/intel-rapl:*/*/energy_uj"):
        try:
            with open(p, "r") as f:
                total_uj += int(f.read().strip())
        except Exception:
            pass
    return total_uj / 1e6

# optional small wrapper to run perf stat for interval and parse misses (reuse your earlier parser idea)
# def perf_stat_one_shot(pid, interval):
#     try:
#         cmd = ['perf', 'stat', '-e', 'cache-references,cache-misses', '-p', str(pid), '--', 'sleep', str(interval)]
#         p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=interval+5)
#         out = (p.stderr or "") + "\n" + (p.stdout or "")
#         # find cache-misses token using simple regex: last numeric right before 'cache-misses'
#         import re
#         m = re.search(r'([0-9][0-9,]*)\s+cache-misses', out)
#         if m:
#             return float(m.group(1).replace(',', ''))
#     except Exception:
#         pass
#     return None

# sampler thread used during a run to collect psutil samples
def sample_in_background(stop_event, interval, samples):
    proc = psutil.Process(os.getpid())
    while not stop_event.is_set():
        t = time.time()
        cpu = psutil.cpu_percent(interval=None)
        p_cpu = proc.cpu_percent(interval=None) / (psutil.cpu_count(logical=True) or 1)
        mem = psutil.virtual_memory().used
        rss = proc.memory_info().rss
        samples.append({"t": t, "cpu": cpu, "proc_cpu": p_cpu, "mem": mem, "rss": rss})
        time.sleep(interval)

# run one experiment configuration and return aggregated stats
def run_one_experiment(city, vals, model_name, context_days, horizon_hours, pipeline, repeat_windows_step=None):
    # convert days->hours
    context_hours = int(context_days) * 24
    pred_len = int(horizon_hours)
    # start background samplers
    stop_event = threading.Event()
    samples = []
    sampler = threading.Thread(target=sample_in_background, args=(stop_event, SAMPLE_INTERVAL, samples), daemon=True)
    sampler.start()

    # record energy start
    try:
        energy_before = read_rapl_joules()
    except Exception:
        energy_before = None

    # optionally run perf stat for this process during the run in parallel (cheap one-shot)
    perf_misses = None
    if USE_PERF:
        # spawn perf stat as separate short run; we run it in parallel and keep it running for roughly the same duration
        perf_proc = None
        try:
            perf_proc = subprocess.Popen(['perf','stat','-e','cache-references,cache-misses','-p', str(os.getpid()), '--','sleep', str( max(5, SAMPLE_INTERVAL*10) )],
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except Exception:
            perf_proc = None

    # actual evaluation (reuse your evaluate_series_zero_shot implementation)
    rmses = evaluate_series_zero_shot(vals, context_hours, pred_len, pipeline, step=repeat_windows_step)
    # stop samplers
    stop_event.set()
    sampler.join(timeout=2)

    # get energy after
    try:
        energy_after = read_rapl_joules()
    except Exception:
        energy_after = None

    # parse perf_proc output if available
    if perf_proc:
        try:
            stderr, stdout = perf_proc.communicate(timeout=5)
            import re
            m = re.search(r'([0-9][0-9,]*)\s+cache-misses', stderr or "")
            if m:
                perf_misses = float(m.group(1).replace(',', ''))
        except Exception:
            pass

    # aggregate samples
    avg_cpu = float(np.mean([s["cpu"] for s in samples])) if samples else float("nan")
    avg_proc_rss = float(np.mean([s["rss"] for s in samples])) if samples else float("nan")

    avg_rmse = float(np.mean(rmses)) if len(rmses) > 0 else float("nan")
    std_rmse = float(np.std(rmses)) if len(rmses) > 0 else float("nan")
    energy_delta = None
    if energy_before is not None and energy_after is not None:
        energy_delta = float(energy_after - energy_before)

    return {
        "city": city,
        "model": model_name,
        "context_days": context_days,
        "horizon_hours": horizon_hours,
        "avg_rmse": avg_rmse,
        "std_rmse": std_rmse,
        "avg_cpu": avg_cpu,
        "avg_rss": avg_proc_rss,
        "energy_j": energy_delta,
        "perf_cache_misses": perf_misses,
        "n_windows": len(rmses),
    }

# Example orchestration (main)
def main():
    # start metrics exporter for this script
    start_http_server(EXPORTER_PORT)
    print("Chronos experiment exporter on port", EXPORTER_PORT)

    # load pipelines & data (reuse your code: load_pipeline, read_city_file)
    pipeline = load_pipeline(MODEL_NAME, DEVICE)

    # load city series
    all_city_series = {}
    for city, fname in CITY_FILES.items():
        ts, vals = read_city_file(fname, resample_hourly=RESAMPLE_HOURLY)
        all_city_series[city] = (ts, vals)

    # CSV log
    csv_fn = Path(OUT_DIR) / "chronos_experiment_results.csv"
    write_header = not csv_fn.exists()
    with open(csv_fn, "a", newline="") as cf:
        writer = csv.writer(cf)
        if write_header:
            writer.writerow([
                "city","model","context_days","horizon_hours","avg_rmse","std_rmse",
                "avg_cpu_pct","avg_rss_bytes","energy_j","perf_cache_misses","n_windows","timestamp"])
        # run sweeps (example: contexts list and horizons list)
        for city, (ts, vals) in all_city_series.items():
            for model in [MODEL_NAME]:   # extend with model variants if you have them
                for ctx_days in CONTEXT_DAYS_LIST:  
                    # horizon fixed 24 for first sweep
                    res = run_one_experiment(city, vals, model, ctx_days, 24, pipeline)
                    # set Prometheus gauges (labels strings)
                    labels = {"city": city, "model": model.replace("/", "_"), "context_days": str(ctx_days), "horizon_hours": "24"}
                    RMSE_G.labels(**labels).set(res["avg_rmse"] if res["avg_rmse"]==res["avg_rmse"] else 0.0)
                    CPU_AVG_G.labels(**labels).set(res["avg_cpu"])
                    RSS_AVG_G.labels(**labels).set(res["avg_rss"])
                    if res["energy_j"] is not None:
                        ENERGY_DELTA_G.labels(**labels).set(res["energy_j"])
                    if res["perf_cache_misses"] is not None:
                        CACHE_MISSES_G.labels(**labels).set(res["perf_cache_misses"])
                    writer.writerow([res["city"], res["model"], res["context_days"], res["horizon_hours"],
                                     res["avg_rmse"], res["std_rmse"], res["avg_cpu"], res["avg_rss"],
                                     res["energy_j"], res["perf_cache_misses"], res["n_windows"], time.time()])
                    cf.flush()
                    print("wrote row for", labels)

        # second sweep: fix context 10 days and vary horizons
        for city,(ts,vals) in all_city_series.items():
            for model in [MODEL_NAME]:
                for horizon in HORIZONS_HOURS_LIST:
                    res = run_one_experiment(city, vals, model, 10, horizon, pipeline)
                    labels = {"city": city, "model": model.replace("/", "_"), "context_days": "10", "horizon_hours": str(horizon)}
                    RMSE_G.labels(**labels).set(res["avg_rmse"] if res["avg_rmse"]==res["avg_rmse"] else 0.0)
                    CPU_AVG_G.labels(**labels).set(res["avg_cpu"])
                    RSS_AVG_G.labels(**labels).set(res["avg_rss"])
                    if res["energy_j"] is not None:
                        ENERGY_DELTA_G.labels(**labels).set(res["energy_j"])
                    if res["perf_cache_misses"] is not None:
                        CACHE_MISSES_G.labels(**labels).set(res["perf_cache_misses"])
                    writer.writerow([res["city"], res["model"], res["context_days"], res["horizon_hours"],
                                     res["avg_rmse"], res["std_rmse"], res["avg_cpu"], res["avg_rss"],
                                     res["energy_j"], res["perf_cache_misses"], res["n_windows"], time.time()])
                    cf.flush()
                    print("wrote row for", labels)

    print("All experiments finished. CSV at", csv_fn)

if __name__ == "__main__":
    main()
