from llama_cpp import Llama
import time, json, os, sys
import psutil, platform
from pathlib import Path
import statistics
import subprocess
import csv

HOME = Path.home()
WORK = HOME / "llm-local"
PROMPTS = WORK / "prompts.txt"
RESULT_CSV = WORK / "results_summary.csv"
INFER_COUNT = 2
MAX_TOKENS = 128
THREADS = os.cpu_count() or 1
MODEL_FILES = [
    "model.gguf", 
    "mistral-7b-v0.1.Q5_K_M.gguf", 
    "llama-2-13b.Q4_K_M.gguf",
]

def hardware_report():
    print("=== Hardware / Environment ===")
    print("Platform:", platform.system(), platform.release(), platform.version())
    print("Platform node:", platform.node())
    try:
        cpuinfo = ""
        p = subprocess.run(["lscpu"], capture_output=True, text=True)
        if p.returncode == 0:
            cpuinfo = p.stdout.splitlines()
            for line in cpuinfo:
                if line.lower().startswith("model name") or line.lower().startswith("cpu(s):") or line.lower().startswith("thread(s) per core"):
                    print(line)
    except Exception:
        pass
    print("python:", sys.executable, sys.version.splitlines()[0])
    print("logical CPUs (os):", os.cpu_count())
    print("physical cores (psutil):", psutil.cpu_count(logical=False))
    vm = psutil.virtual_memory()
    print("total RAM (GiB):", round(vm.total / (1024**3), 2))
    print("==============================\n")

def load_prompts():
    if not PROMPTS.exists():
        print("Prompts file not found at", PROMPTS)
        sys.exit(1)
    lines = [l.strip() for l in PROMPTS.read_text().splitlines() if l.strip()]
    if not lines:
        print("prompts.txt is empty")
        sys.exit(1)
    out = []
    idx = 0
    while len(out) < INFER_COUNT:
        out.append(lines[idx % len(lines)])
        idx += 1
    return out

def run_model(model_path):
    modelname = Path(model_path).name
    print(f"\n--- Model: {modelname} ---")
    if not Path(model_path).exists():
        print("SKIP (file not found):", model_path)
        return {"model": modelname, "status": "missing"}
    print("loading model (this may take a moment)...")
    llm = Llama(model_path=str(model_path))
    prompts = load_prompts()
    latencies = []
    out_fn = WORK / f"outputs_{modelname}.jsonl"
    with open(out_fn, "w") as out:
        for i,p in enumerate(prompts):
            t0 = time.time()
            res = llm.create_completion(prompt=p, max_tokens=MAX_TOKENS)
            t1 = time.time()
            latency = t1 - t0
            latencies.append(latency)
            text = ""
            try:
                text = res["choices"][0]["text"]
            except Exception:
                text = str(res)
            out.write(json.dumps({"i": i, "prompt": p, "latency": latency, "output": text}) + "\n")
            if (i+1) % 10 == 0:
                print(f"  completed {i+1}/{len(prompts)} inferences")
    avg = statistics.mean(latencies) if latencies else None
    sd = statistics.pstdev(latencies) if latencies else None
    print(f"Model {modelname} avg latency over {len(latencies)} runs: {avg:.4f}s (std {sd:.4f}s)")
    return {"model": modelname, "status": "done", "avg_latency_s": avg, "std_latency_s": sd, "runs": len(latencies)}

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    hardware_report()
    resolved = [WORK / p for p in MODEL_FILES]
    results = []
    for mp in resolved:
        r = run_model(mp)
        results.append(r)
    with open(RESULT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model","status","avg_latency_s","std_latency_s","runs"])
        for r in results:
            w.writerow([r.get("model"), r.get("status"), r.get("avg_latency_s"), r.get("std_latency_s"), r.get("runs")])
    print("\nWrote summary to", RESULT_CSV)
    print("Per-inference outputs are in files named outputs_<modelfilename>.jsonl in", WORK)

if __name__ == "__main__":
    main()

