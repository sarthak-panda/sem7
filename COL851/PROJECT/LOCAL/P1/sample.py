# chronos_zero_shot_pm_forecast.py
# Zero-shot PM2.5 forecasting with Amazon Chronos / Chronos-Bolt.
# - Loads two csv time-series (gurgaon.csv, patna.csv)
# - Uses only timestamp (col 1) and PM (col 5)
# - Runs zero-shot forecasts with pre-trained Chronos( Bolt ) models on CPU
# - Experiments with different context lengths and horizons
# - Produces two key plots required by the assignment and saves numeric results

"""
Requirements (install before running):
    pip install git+https://github.com/amazon-science/chronos-forecasting.git
    pip install pandas numpy matplotlib tqdm scikit-learn
    # torch will be installed as a dependency but ensure a CPU build is available

Usage:
    python chronos_zero_shot_pm_forecast.py --files gurgaon.csv patna.csv

Outputs (saved to current directory):
    results_<city>_<model>.csv  # RMSE results table
    plot_context_vs_rmse_<city>_<model>.png  # 24h horizon, various context lengths (biggest model)
    plot_horizon_vs_rmse_<city>_<model>.png   # 10 day context, various horizons

Notes:
 - The script attempts to load the largest model listed in models_to_try and will fall back
   to smaller variants if loading fails on your machine.
 - The evaluation uses a sliding-window walk-forward strategy over the last `test_window_days` days
   to compute average RMSEs.
"""

import argparse
import math
import os
import sys
import warnings
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import matplotlib.pyplot as plt

# Chronos import
try:
    from chronos import BaseChronosPipeline
except Exception as e:
    raise ImportError(
        "Could not import chronos. Please install it via:\n"
        "pip install git+https://github.com/amazon-science/chronos-forecasting.git"
    )


# ------------------------ Utilities ------------------------

def detect_time_and_value_columns(df: pd.DataFrame):
    """Try to automatically detect timestamp and PM columns.
    Returns (time_col, value_col).
    """
    # common timestamp names
    time_candidates = [c for c in df.columns if c.lower().strip() in [
        'from date', 'from_date', 'timestamp', 'time', 'date', 'datetime', 'fromdate'
    ]]
    if len(time_candidates) == 0:
        # fallback: first column
        time_col = df.columns[0]
    else:
        time_col = time_candidates[0]

    # PM column candidates - the user indicated 'calibPM' likely is present or 5th column
    value_candidates = [c for c in df.columns if 'pm' in c.lower() or 'calib' in c.lower() or 'calibpm' in c.lower()]
    if len(value_candidates) == 0:
        # fallback: 5th column if present else last
        if df.shape[1] >= 5:
            value_col = df.columns[4]
        else:
            value_col = df.columns[-1]
    else:
        value_col = value_candidates[0]

    return time_col, value_col


def load_series(path: str):
    df = pd.read_csv(path)
    time_col, value_col = detect_time_and_value_columns(df)
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    series = df[[time_col, value_col]].copy()
    series.columns = ['time', 'value']
    # handle missing values: forward fill then backward fill
    series['value'] = pd.to_numeric(series['value'], errors='coerce')
    series['value'] = series['value'].ffill().bfill()
    return series


# ------------------------ Chronos wrapper ------------------------

def try_load_pipeline(models_to_try: List[str]):
    """Try loading models in order, return (model_name, pipeline) of first successful load."""
    last_err = None
    for model_name in models_to_try:
        try:
            print(f"Trying to load model {model_name} on CPU ...")
            pipeline = BaseChronosPipeline.from_pretrained(model_name, device_map='cpu')
            print(f"Loaded model: {model_name}")
            return model_name, pipeline
        except Exception as e:
            last_err = e
            warnings.warn(f"Could not load {model_name}: {e}")
            continue
    raise RuntimeError(f"Failed to load any model from the list. Last error: {last_err}")


def safe_predict(pipeline, context_array: np.ndarray, prediction_length: int, num_samples: int = None):
    """Call pipeline.predict robustly for both Chronos and Chronos-Bolt variants.

    - Converts the numpy context to a torch tensor and calls predict with keyword `context=` as
      recommended in the Chronos README. If `num_samples` is not accepted by the pipeline,
      the function will retry without it.
    - Handles common return shapes:
        * (batch, num_samples, pred_len) -> mean over samples
        * (batch, num_quantiles, pred_len) -> mean over quantiles (approximate point forecast)
        * (batch, pred_len) -> direct point forecast

    Returns a numpy array with shape (batch, pred_len).
    """
    import traceback
    try:
        import torch
    except Exception:
        raise ImportError("torch is required to run Chronos predictions. Please install a CPU build of torch.")

    # convert to torch tensor with float32
    if isinstance(context_array, np.ndarray):
        ctx_tensor = torch.tensor(context_array, dtype=torch.float32)
    else:
        ctx_tensor = context_array

    # ensure shape: either 1D tensor (seq,) or 2D (batch, seq).
    if ctx_tensor.ndim == 1:
        ctx_arg = ctx_tensor
    else:
        # left-padded 2D tensor expected (batch, seq)
        ctx_arg = ctx_tensor

    # try calling predict with and without num_samples
    last_exc = None
    tried_kwargs = []
    for use_num_samples in ([True] if num_samples else []) + [False]:
        kwargs = dict(prediction_length=prediction_length)
        if use_num_samples:
            kwargs['num_samples'] = num_samples
        tried_kwargs.append(kwargs)
        try:
            out = pipeline.predict(context=ctx_arg, **kwargs)
            out_np = np.asarray(out)
            # normalize to (batch, pred_len)
            if out_np.ndim == 3:
                # (batch, samples_or_quantiles, pred_len) -> mean across axis=1
                out_point = out_np.mean(axis=1)
            elif out_np.ndim == 2:
                out_point = out_np
            else:
                raise ValueError(f"Unexpected prediction output shape: {out_np.shape}")
            return out_point
        except TypeError as e:
            # method doesn't accept the provided kwarg combination, retry
            last_exc = e
            continue
        except Exception as e:
            # bubble up runtime errors with trace for easier debugging
            tb = traceback.format_exc()
            raise RuntimeError(f"pipeline.predict failed: {e}{tb}")

    # if we get here all attempts failed due to TypeError
    raise TypeError(f"pipeline.predict failed with tried kwargs: {tried_kwargs}. Last error: {last_exc}")


# ------------------------ Evaluation ------------------------

def evaluate_zero_shot(
    pipeline,
    series: pd.DataFrame,
    context_length_hours: int,
    horizon_hours: int,
    test_window_days: int = 60,
    num_samples: int = 20,
    step_hours: int = None,
):
    """
    Sliding-window walk-forward zero-shot evaluation.
    - context_length_hours: how many past hours are fed to the model
    - horizon_hours: forecast horizon
    - test_window_days: how many most recent days to use for evaluation
    - step_hours: step between windows (defaults to horizon_hours)

    Returns average RMSE across windows and list of per-window RMSEs.
    """
    step_hours = step_hours or horizon_hours
    values = series['value'].values.astype(float)
    length = len(values)
    test_len = min(test_window_days * 24, length)

    # We will run windows that end inside the last `test_len` observations.
    # Valid last context_end index i must satisfy: i + horizon <= length and i - context_length >=0
    end_idx = length - horizon_hours
    start_eval_idx = max(context_length_hours, length - test_len)

    rmses = []
    idxs = list(range(start_eval_idx, end_idx + 1, step_hours))
    if len(idxs) == 0:
        raise ValueError("Not enough data for the chosen context/horizon/test window settings.")

    for i in tqdm(idxs, desc=f"Eval ctx={context_length_hours}h horizon={horizon_hours}h", leave=False):
        context = values[i - context_length_hours:i]
        target = values[i:i + horizon_hours]
        # format context for pipeline: shape [batch, seq_len]
        context_array = context.reshape(1, -1)
        # call predict
        try:
            pred_raw = safe_predict(pipeline, context_array, prediction_length=horizon_hours, num_samples=num_samples)
        except Exception as e:
            raise RuntimeError(f"Prediction failed for window ending at idx {i}: {e}")

        # turn outputs into mean prediction across samples
        # possible shapes: (batch, num_samples, pred_len) or (batch, pred_len)
        if pred_raw.ndim == 3:
            # mean across samples
            pred_mean = pred_raw.mean(axis=1)[0]
        elif pred_raw.ndim == 2:
            pred_mean = pred_raw[0]
        else:
            raise ValueError(f"unexpected prediction shape: {pred_raw.shape}")

        # compute RMSE for this window
        rmse = math.sqrt(mean_squared_error(target, pred_mean))
        rmses.append(rmse)

    avg_rmse = float(np.mean(rmses))
    return avg_rmse, rmses


# ------------------------ Orchestration & plotting ------------------------

def run_experiments_for_city(
    pipeline,
    model_name: str,
    series: pd.DataFrame,
    city_name: str,
    context_days_list: List[int],
    horizons_hours_list: List[int],
    biggest_model_for_context_plot: bool = False,
    num_samples: int = 20,
    test_window_days: int = 60,
):
    results = []

    # Experiment A: fixed horizon = 24h, vary context lengths (days)
    fixed_horizon = 24
    ctx_results = {}
    print(f"\nRunning context-length sweep for city={city_name}, model={model_name}")
    for ctx_days in context_days_list:
        ctx_hours = ctx_days * 24
        try:
            avg_rmse, rmses = evaluate_zero_shot(
                pipeline,
                series,
                context_length_hours=ctx_hours,
                horizon_hours=fixed_horizon,
                test_window_days=test_window_days,
                num_samples=num_samples,
                step_hours=fixed_horizon,
            )
        except Exception as e:
            warnings.warn(f"Failed ctx {ctx_days}d: {e}")
            avg_rmse = float('nan')
            rmses = []
        ctx_results[ctx_days] = dict(avg_rmse=avg_rmse, rmses=rmses)
        results.append((city_name, model_name, 'context_sweep', ctx_days, fixed_horizon, avg_rmse))

    # Experiment B: fixed context = 10 days, vary horizons
    fixed_context_days = 10
    fixed_context_hours = fixed_context_days * 24
    horizon_results = {}
    print(f"\nRunning horizon sweep for city={city_name}, model={model_name}")
    for horizon in horizons_hours_list:
        try:
            avg_rmse, rmses = evaluate_zero_shot(
                pipeline,
                series,
                context_length_hours=fixed_context_hours,
                horizon_hours=horizon,
                test_window_days=test_window_days,
                num_samples=num_samples,
                step_hours=horizon,
            )
        except Exception as e:
            warnings.warn(f"Failed horizon {horizon}h: {e}")
            avg_rmse = float('nan')
            rmses = []
        horizon_results[horizon] = dict(avg_rmse=avg_rmse, rmses=rmses)
        results.append((city_name, model_name, 'horizon_sweep', fixed_context_days, horizon, avg_rmse))

    # Save results CSV
    rows = []
    for ctx_days, v in ctx_results.items():
        rows.append({
            'city': city_name,
            'model': model_name,
            'experiment': 'context_sweep',
            'context_days': ctx_days,
            'horizon_hours': fixed_horizon,
            'avg_rmse': v['avg_rmse']
        })
    for horizon, v in horizon_results.items():
        rows.append({
            'city': city_name,
            'model': model_name,
            'experiment': 'horizon_sweep',
            'context_days': fixed_context_days,
            'horizon_hours': horizon,
            'avg_rmse': v['avg_rmse']
        })
    out_df = pd.DataFrame(rows)
    csv_name = f"results_{city_name.lower()}_{model_name.replace('/', '_')}.csv"
    out_df.to_csv(csv_name, index=False)
    print(f"Saved results to {csv_name}")

    # Plot A: for 24h horizon, avg RMSE vs context lengths
    try:
        ctx_x = sorted(ctx_results.keys())
        ctx_y = [ctx_results[k]['avg_rmse'] for k in ctx_x]
        plt.figure(figsize=(8, 5))
        plt.plot(ctx_x, ctx_y, marker='o')
        plt.xlabel('Context length (days)')
        plt.ylabel(f'Average RMSE (horizon={fixed_horizon}h)')
        plt.title(f'{city_name} — Avg RMSE vs Context Length ({model_name})')
        plt.grid(True)
        plot_name = f"plot_context_vs_rmse_{city_name.lower()}_{model_name.replace('/', '_')}.png"
        plt.savefig(plot_name, dpi=200)
        plt.close()
        print(f"Saved plot {plot_name}")
    except Exception as e:
        warnings.warn(f"Failed to make context plot: {e}")

    # Plot B: for context=10 days, avg RMSE vs horizon hours
    try:
        h_x = sorted(horizon_results.keys())
        h_y = [horizon_results[k]['avg_rmse'] for k in h_x]
        plt.figure(figsize=(8, 5))
        plt.plot(h_x, h_y, marker='o')
        plt.xlabel('Forecast horizon (hours)')
        plt.ylabel(f'Average RMSE (context={fixed_context_days} days)')
        plt.title(f'{city_name} — Avg RMSE vs Horizon ({model_name})')
        plt.grid(True)
        plot_name = f"plot_horizon_vs_rmse_{city_name.lower()}_{model_name.replace('/', '_')}.png"
        plt.savefig(plot_name, dpi=200)
        plt.close()
        print(f"Saved plot {plot_name}")
    except Exception as e:
        warnings.warn(f"Failed to make horizon plot: {e}")

    return out_df


# ------------------------ Main ------------------------

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', required=True, help='CSV file paths for the cities (gurgaon.csv patna.csv)')
    parser.add_argument('--models', nargs='+', default=[
        'amazon/chronos-bolt-base',
        'amazon/chronos-bolt-small',
        'amazon/chronos-bolt-tiny',
    ], help='Chronos model names to try in descending compute order')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of Monte Carlo samples for predictive distribution')
    parser.add_argument('--test_window_days', type=int, default=60, help='How many last days to evaluate over (sliding windows)')
    args = parser.parse_args(argv)

    # load series files
    city_series = {}
    for f in args.files:
        city_name = os.path.splitext(os.path.basename(f))[0]
        print(f"Loading {f} as city {city_name} ...")
        s = load_series(f)
        city_series[city_name] = s
        print(f"Loaded {len(s)} rows for {city_name}")

    # try to load model pipeline
    model_name, pipeline = try_load_pipeline(args.models)

    # experiment settings from assignment
    context_days_list = [2, 4, 8, 10, 14]  # days
    horizons_hours_list = [4, 8, 12, 24, 48]

    # run for each city
    for city_name, series in city_series.items():
        print(f"\n=== Running experiments for {city_name} ===")
        out_df = run_experiments_for_city(
            pipeline=pipeline,
            model_name=model_name,
            series=series,
            city_name=city_name,
            context_days_list=context_days_list,
            horizons_hours_list=horizons_hours_list,
            num_samples=args.num_samples,
            test_window_days=args.test_window_days,
        )
        print(out_df)


if __name__ == '__main__':
    main()
