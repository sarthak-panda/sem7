#!/usr/bin/env python3
"""
inference_service.py

Usage:
  # start service (load model once, expose metrics + small HTTP endpoint)
  python inference_service.py --model ./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --port 8000

  # run benchmark for 100 synchronous runs (loads model once then runs N times)
  python inference_service.py --model ./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --benchmark 100 --prompt "Hello"
"""

import argparse
import time
import csv
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from prometheus_client import start_http_server, Histogram, Counter, Gauge
import json
from llama_cpp import Llama

# Prometheus metrics
LATENCY = Histogram('inference_latency_seconds', 'LLM inference latency (seconds)', buckets=(0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5))
REQS = Counter('inference_requests_total', 'Total inference requests')
IN_FLIGHT = Gauge('inference_in_flight', 'Number of in-flight inference calls')
LAST_LATENCY = Gauge('inference_last_latency_seconds', 'Last inference latency (seconds)')

# === SYSTEM METRICS EXTENSION (drop into your inference_service.py) ===
import psutil
import threading
import time
import subprocess
import re
import glob
import os

# Prometheus metrics (add these along with your existing metrics)
CPU_UTIL = Gauge('system_cpu_percent', 'System CPU utilization percent')
CPU_PROCESS_UTIL = Gauge('process_cpu_percent', 'This process CPU percent')
CPU_FREQ = Gauge('system_cpu_freq_mhz', 'Current CPU frequency (MHz)')
MEM_USED = Gauge('system_memory_used_bytes', 'System memory used (bytes)')
MEM_AVAILABLE = Gauge('system_memory_available_bytes', 'System memory available (bytes)')
TEMP = Gauge('system_cpu_temperature_celsius', 'CPU temperature (Celsius)')
POWER_WATTS = Gauge('system_power_watts', 'System package power (Watts)')
CACHE_MISSES_PER_SEC = Gauge('proc_cache_misses_per_sec', 'Cache misses per second for process (approx)')
CACHE_REFERENCES_PER_SEC = Gauge('proc_cache_references_per_sec', 'Cache references per second for process (approx)')
INFER_THROUGHPUT_GAUGE = Gauge("llm_inference_throughput_ips", "Instantaneous inference throughput (inferences/sec)")

_METRICS_SAMPLER_INTERVAL = 1.0  # seconds

def _read_rapl_energy_uj_paths():
    """Find RAPL energy_uj files (Linux)"""
    candidates = []
    for p in glob.glob('/sys/class/powercap/intel-rapl:*/*/energy_uj') + glob.glob('/sys/class/powercap/*/*/energy_uj'):
        try:
            # ensure readable
            open(p).close()
            candidates.append(p)
        except Exception:
            continue
    return candidates

_RAPL_PATHS = _read_rapl_energy_uj_paths()

def read_total_energy_uj():
    """Sum energy_uj across rapl domains (returns int or None)."""
    if not _RAPL_PATHS:
        return None
    total = 0
    for p in _RAPL_PATHS:
        try:
            with open(p, 'r') as f:
                total += int(f.read().strip())
        except Exception:
            return None
    return total

def _parse_perf_stat_output(out):
    # parse numbers for cache-misses and cache-references (robust to commas)
    def find_count(name):
        m = re.search(r'([\d,]+)\s+.*%s' % re.escape(name), out)
        if m:
            return int(m.group(1).replace(',', ''))
        # fallback: search lines
        m = re.search(r'([\d,]+)\s+%s' % re.escape(name), out)
        if m:
            return int(m.group(1).replace(',', ''))
        return None
    misses = find_count('cache-misses')
    refs = find_count('cache-references')
    return refs, misses

def _perf_sample_once(pid, interval_s=_METRICS_SAMPLER_INTERVAL):
    """Run perf stat for a short interval attached to pid. Requires perf to be installed and proper permissions."""
    try:
        # Call perf stat for a short sleep; output is in stderr usually
        cmd = ['perf', 'stat', '-e', 'cache-references,cache-misses', '-p', str(pid), '--', 'sleep', str(interval_s)]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=interval_s+5)
        out = p.stderr + "\n" + p.stdout
        refs, misses = _parse_perf_stat_output(out)
        return refs, misses
    except FileNotFoundError:
        return None, None
    except subprocess.SubprocessError:
        return None, None

def system_metrics_collector(stop_event, sampling_interval=_METRICS_SAMPLER_INTERVAL):
    proc = psutil.Process(os.getpid())
    # prime cpu_percent
    psutil.cpu_percent(interval=None)
    proc.cpu_percent(interval=None)

    # for power calculation via RAPL
    prev_energy = read_total_energy_uj()
    prev_time = time.monotonic() if prev_energy is not None else None

    # perf baseline (optional)
    perf_available = True
    try:
        subprocess.run(['perf', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        perf_available = False

    # main loop
    while not stop_event.is_set():
        t0 = time.monotonic()

        # CPU and process CPU %
        CPU_UTIL.set(psutil.cpu_percent(interval=None))
        CPU_PROCESS_UTIL.set(proc.cpu_percent(interval=None) / psutil.cpu_count(logical=True))  # normalized

        # CPU freq (current)
        try:
            cf = psutil.cpu_freq()
            if cf and cf.current:
                CPU_FREQ.set(cf.current)
        except Exception:
            pass

        # memory
        vm = psutil.virtual_memory()
        MEM_USED.set(vm.used)
        MEM_AVAILABLE.set(vm.available)

        # temperature (pick first available CPU sensor)
        try:
            temps = psutil.sensors_temperatures()
            # pick any entry with 'core' or 'package' or first
            value = None
            for k, v in temps.items():
                if not v:
                    continue
                # v is list of namedtuples; pick highest current
                value = max([x.current for x in v if getattr(x, 'current', None) is not None])
                if value is not None:
                    break
            if value is not None:
                TEMP.set(value)
        except Exception:
            pass

        # power via RAPL
        try:
            cur_energy = read_total_energy_uj()
            if cur_energy is not None and prev_energy is not None:
                now = time.monotonic()
                dt = now - prev_time if prev_time else sampling_interval
                # energy_uj is microjoules
                watts = ((cur_energy - prev_energy) / 1e6) / max(dt, 1e-6)
                # sometimes counters wrap; handle negative
                if watts < 0:
                    watts = None
                else:
                    POWER_WATTS.set(watts)
                prev_energy = cur_energy
                prev_time = now
            elif cur_energy is not None:
                prev_energy = cur_energy
                prev_time = time.monotonic()
        except Exception:
            pass

        # cache metrics via perf (may be slow / require permissions)
        if perf_available:
            refs, misses = _perf_sample_once(proc.pid, interval_s=sampling_interval)
            # perf already slept for interval, so no need to sleep again
            if refs is not None:
                CACHE_REFERENCES_PER_SEC.set(refs / sampling_interval)
            if misses is not None:
                CACHE_MISSES_PER_SEC.set(misses / sampling_interval)
            # continue to next iteration (perf consumed the interval)
            continue

        # if perf not available, just sleep the sampling interval
        elapsed = time.monotonic() - t0
        to_sleep = max(0.0, sampling_interval - elapsed)
        time.sleep(to_sleep)

def start_system_metrics_collector(interval=_METRICS_SAMPLER_INTERVAL):
    stop_event = threading.Event()
    t = threading.Thread(target=system_metrics_collector, args=(stop_event, interval), daemon=True)
    t.start()
    return stop_event

# === end of extension ===



import os
import json

def run_from_file(llm, infile, out_jsonl="outputs.jsonl", out_csv="latencies.csv",
                  max_tokens=256, temperature=0.7, sleep_between=0.0):
    """
    Run prompts from `infile` (one prompt per line), write results to jsonl and csv.
    Exposes Prometheus metrics because start_http_server(...) should be active in main().
    """
    if not os.path.exists(infile):
        print(f"[!] infile not found: {infile}")
        return

    print(f"[+] Running prompts from: {infile}")
    total = 0.0
    count = 0
    with open(infile, "r", encoding="utf-8") as inf, \
         open(out_jsonl, "w", encoding="utf-8") as outj, \
         open(out_csv, "w", newline="", encoding="utf-8") as outc:

        import csv as _csv
        writer = _csv.writer(outc)
        writer.writerow(["run", "latency_s"])

        for i, line in enumerate(inf, start=1):
            prompt = line.strip()
            if not prompt:
                continue
            count += 1
            try:
                latency, text, raw = single_inference(llm, prompt, max_tokens=max_tokens, temperature=temperature)
                total += latency
                # write JSONL
                rec = {"id": i, "latency": round(latency, 6), "prompt": prompt, "output": text}
                outj.write(json.dumps(rec) + "\n")
                outj.flush()
                # write CSV line
                writer.writerow([i, f"{latency:.6f}"])
                outc.flush()
                print(f"[{i}] latency={latency:.3f}s  prompt_len={len(prompt)}")
            except Exception as e:
                print(f"[!] Error at prompt {i}: {e}")
                # write error line
                outj.write(json.dumps({"id": i, "latency": None, "prompt": prompt, "output": "", "error": str(e)}) + "\n")
                outj.flush()
                writer.writerow([i, "error"])
                outc.flush()

            if sleep_between and sleep_between > 0:
                time.sleep(sleep_between)

    # summary
    import statistics
    print("---- File run summary ----")
    print(f"Prompts processed (lines read): {count}")
    if count > 0:
        mean = total / count
        print(f"Total time (sum latencies): {total:.3f}s  Mean latency: {mean:.3f}s  Throughput: {count / total:.3f} req/s")
    print(f"Outputs => JSONL: {out_jsonl} , CSV: {out_csv}")


def load_model(model_path, n_ctx=2048, n_batch=8, verbose=False):
    print(f"Loading model from: {model_path} ...")
    # chat_format can be adjusted (e.g., "chatml" or "llama-2"), but default works for many chat ggufs.
    llm = Llama(model_path=str(model_path), n_ctx=n_ctx, n_batch=n_batch, verbose=verbose)
    print("Model loaded.")
    return llm

def single_inference(llm, prompt, system_prompt=None, timeout=None, max_tokens=256, temperature=0.7):
    """
    Run a single chat-style inference and measure latency.
    Returns (latency_seconds, text_output, raw_response)
    """
    REQS.inc()
    IN_FLIGHT.inc()
    t0 = time.monotonic()
    try:
        messages = []
        if system_prompt:
            messages.append({"role":"system","content":system_prompt})
        messages.append({"role":"user","content":prompt})

        # create_chat_completion returns a dict; adjust args if you want streaming
        resp = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        # typical shape: resp['choices'][0]['message']['content']
        text = resp['choices'][0]['message']['content']
    finally:
        latency = time.monotonic() - t0
        LAST_LATENCY.set(latency)
        LATENCY.observe(latency)
        IN_FLIGHT.dec()
        if latency > 0:
            INFER_THROUGHPUT_GAUGE.set(1.0 / latency)
    return latency, text, resp

# Small HTTP server to accept requests
class Handler(BaseHTTPRequestHandler):
    server_version = "TinyLlama-Serv/0.1"
    def do_POST(self):
        if self.path != "/generate":
            self.send_response(404); self.end_headers(); return
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8') if content_length else ''
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {}
        prompt = payload.get("prompt", "Hello")
        system = payload.get("system", None)
        max_tokens = int(payload.get("max_tokens", 256))
        temp = float(payload.get("temperature", 0.7))

        latency, text, raw = single_inference(self.server.llm, prompt, system_prompt=system, max_tokens=max_tokens, temperature=temp)
        out = {"latency_s": latency, "text": text}
        data = json.dumps(out).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

def run_server(llm, host, port):
    # attach llm to server so handler can use it
    server = HTTPServer((host, port), Handler)
    server.llm = llm
    print(f"Serving /generate and Prometheus metrics on port {port} (metrics on /metrics via prometheus_client)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down server...")
        server.shutdown()

def run_benchmark(llm, niter, prompt, system_prompt=None, csv_out="latencies.csv", max_tokens=256, temperature=0.7, concurrent=1):
    """
    Run `niter` synchronous inferences (optionally with `concurrent` threads).
    Writes latencies to CSV and prints summary.
    """
    latencies = []
    rows = []
    print(f"Running benchmark: {niter} iterations, concurrency={concurrent}")
    def worker(i):
        latency, text, _ = single_inference(llm, prompt, system_prompt=system_prompt, max_tokens=max_tokens, temperature=temperature)
        rows.append((i, latency))
        latencies.append(latency)

    if concurrent <= 1:
        for i in range(1, niter+1):
            worker(i)
    else:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent) as ex:
            futures = [ex.submit(worker, i) for i in range(1, niter+1)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

    # write CSV
    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "latency_s"])
        for r in sorted(rows, key=lambda x:x[0]):
            writer.writerow([r[0], f"{r[1]:.6f}"])
    # summary
    import statistics
    total = sum(latencies)
    mean = statistics.mean(latencies)
    median = statistics.median(latencies)
    p95 = sorted(latencies)[int(len(latencies)*0.95)-1] if len(latencies)>=20 else max(latencies)
    throughput = len(latencies) / total if total>0 else 0.0
    print(f"Runs: {len(latencies)}  Total time: {total:.3f}s  Mean: {mean:.4f}s  Median: {median:.4f}s  P95: {p95:.4f}s  Throughput: {throughput:.2f} req/s")
    print(f"Latencies saved to {csv_out}")
    return latencies

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .gguf model file")
    ap.add_argument("--port", type=int, default=8000, help="HTTP port for /generate")
    ap.add_argument("--metrics-port", type=int, default=None, help="Prometheus metrics port (defaults to port+1)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--benchmark", type=int, default=0, help="If >0, run N iterations and exit")
    ap.add_argument("--prompt", default="Hello, how are you?", help="Prompt to use for benchmark")
    ap.add_argument("--system", default=None, help="System prompt (optional)")
    ap.add_argument("--csv", default="latencies.csv", help="CSV output for latencies when benchmarking")
    ap.add_argument("--concurrent", type=int, default=1, help="Concurrency for benchmark (threads)")
    ap.add_argument("--n_ctx", type=int, default=2048, help="Context length")
    ap.add_argument("--n_batch", type=int, default=8, help="n_batch for llama-cpp")
    # new file-based args
    ap.add_argument("--infile", default=None, help="Run prompts from this file (one prompt per line) and exit")
    ap.add_argument("--out", default="outputs.jsonl", help="Output JSONL when using --infile")
    ap.add_argument("--out-csv", dest="out_csv", default="latencies.csv", help="Output CSV when using --infile")
    ap.add_argument("--sleep-between", type=float, default=0.0, help="Sleep seconds between requests when using --infile")
    args = ap.parse_args()

    # decide metrics port: default to port+1 if not set
    if args.metrics_port is None:
        metrics_port = args.port + 1
    else:
        metrics_port = args.metrics_port

    # start prometheus client server on a different port so it doesn't clash with /generate
    start_http_server(metrics_port)   # exposes /metrics on metrics_port
    print(f"[+] Prometheus metrics available on http://localhost:{metrics_port}/metrics")

    # load model
    llm = load_model(args.model, n_ctx=args.n_ctx, n_batch=args.n_batch)

    # If infile provided -> run prompts from file and exit
    stop_metrics = start_system_metrics_collector(interval=1.0)

    if args.infile:
        run_from_file(llm, args.infile, out_jsonl=args.out, out_csv=args.out_csv,
                      max_tokens=256, temperature=0.7, sleep_between=args.sleep_between)
        stop_metrics.set()
        return
    

    # benchmark CLI mode (existing)
    if args.benchmark and args.benchmark > 0:
        # direct run (no HTTP)
        run_benchmark(llm, args.benchmark, args.prompt, system_prompt=args.system,
                      csv_out=args.csv, concurrent=args.concurrent)
        stop_metrics.set()
        
        return

    # Normal server mode: run a small HTTP server (it serves /generate).
    run_server(llm, args.host, args.port)


if __name__ == "__main__":
    main()

