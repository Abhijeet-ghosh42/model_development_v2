import argparse
import logging
import math
import os
import re
import warnings

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow_base import MLflowBase

# --- Pre-Script Configuration & Constants ---

# Suppress warnings and setup basic logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directory for input data
DATA_DIR = "data"
# Directory for local artifact output
ARTIFACTS_DIR = "artifacts"

# --- Helper Functions (from Notebook) ---

def extract_info(col_name: str):
    """
    Parse names like 'ch1_7' or 'vh3_12' => (type in {'c','v'}, phase in {1,2,3}, harmonic order int)
    """
    try:
        parts = col_name.split("_")
        type_phase_str = parts[0]
        freq = int(parts[1])
        m = re.match(r"([a-z]+)([0-9]+)", type_phase_str, re.I)
        if not m: return (None, None, None)
        type_str, phase_str = m.groups()
        t = "c" if type_str.lower() == "ch" else "v" if type_str.lower() == "vh" else None
        ph = int(phase_str)
        return (t, ph, freq) if t in {"c", "v"} else (None, None, None)
    except Exception:
        return (None, None, None)

def filter_harmonic_cols(df, t="c", phase=None, order_range=None, parity=None):
    """Filters dataframe columns based on harmonic properties."""
    cols = []
    for col in df.columns:
        tt, ph, fr = extract_info(col)
        if tt != t or fr is None: continue
        if phase is not None and ph != phase: continue
        if order_range is not None:
            lo, hi = order_range
            if not (lo <= fr <= hi): continue
        if parity == "odd" and fr % 2 == 0: continue
        if parity == "even" and fr % 2 != 0: continue
        cols.append(col)
    return cols

def rms_across_harmonics(subdf: pd.DataFrame) -> pd.Series:
    """Calculates the Root Mean Square across rows of a dataframe."""
    if subdf.empty: return pd.Series([], dtype=float)
    vals = subdf.to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        return pd.Series(np.sqrt(np.nanmean(np.square(vals), axis=1)), index=subdf.index)

def build_profile(df, t="c", order_range=None, parity=None):
    """Builds a single harmonic profile by averaging RMS values across phases."""
    phase_series, debug = [], {}
    for ph in [1, 2, 3]:
        cols = filter_harmonic_cols(df, t=t, phase=ph, order_range=order_range, parity=parity)
        debug[f"phase{ph}_ncols"] = len(cols)
        phase_series.append(rms_across_harmonics(df[cols]) if cols else pd.Series(index=df.index, dtype=float))
    
    stacked = pd.concat(phase_series, axis=1) if phase_series else pd.DataFrame(index=df.index)
    profile = stacked.mean(axis=1, skipna=True) if not stacked.empty else pd.Series(index=df.index, dtype=float)
    
    debug["rows"] = int(df.shape[0])
    debug["valid_rows"] = int(profile.dropna().shape[0])
    return profile, debug

def safe_pair(mean_val, std_val):
    """Ensures mean and std are valid floats for YAML output."""
    m = float(mean_val) if not np.isnan(mean_val) else 0.0
    s = float(std_val) if not np.isnan(std_val) else 0.0
    return [round(m, 6), round(s, 6)]

def cohens_d(a, b):
    """Calculates Cohen's d for effect size between two groups."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if a.size < 2 or b.size < 2: return float("nan")
    m1, m2 = a.mean(), b.mean()
    s = np.sqrt(((a.var(ddof=1) + b.var(ddof=1)) / 2.0))
    return float((m2 - m1) / s) if s > 0 else float("nan")

def psi(a, b, bins=10):
    """Calculates the Population Stability Index (PSI)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if a.size < 10 or b.size < 10: return float("nan")
    qs = np.quantile(a, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    val = 0.0
    for i in range(bins):
        p = ((a >= qs[i]) & (a < qs[i + 1])).mean()
        q = ((b >= qs[i]) & (b < qs[i + 1])).mean()
        if p > 0 and q > 0:
            val += (p - q) * np.log(p / q)
    return float(val)

def generate_run_notes(params, metrics, output_yaml):
    """Generates a markdown note for the MLflow run."""
    def _fmt(x, nd=4):
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))): return "NaN"
        try: return f"{float(x):.{nd}f}"
        except: return str(x)

    return f"""
### Run Documentation
This note summarizes the key configurations and metrics for this gearbox fault model training run.

#### 📦 Artifact (the deliverable)
- **Config YAML**: `{output_yaml}`
  This is the file used by the online real-time detector.

#### ⚙️ Parameters (setup & reproducibility)
- **algorithm_name**: {params['run_name']}
- **tenant_id / machine_id**: {params['tenant_id']} / {params['machine_id']}
- **dataset_path**: {params['dataset_filename']}
- **stoppage_current_threshold**: {params['stoppage_current_threshold']}
- **use_parity_profiles**: {params['use_parity_profiles']}
- **eval_baseline_fraction**: {params['baseline_fraction']}
- **eval_aggregation**: {params['aggregation']}
- **eval_threshold_method**: {params['threshold_method']}
- **eval_decision_threshold**: {_fmt(metrics.get('eval_decision_threshold', 'N/A'))}

#### 📊 Core Metrics (for benchmarking without labels)
- **quality_index**: **{_fmt(metrics.get('quality_index', 'N/A'))}**
- **drift_cohens_d**: {_fmt(metrics.get('drift_cohens_d', 'N/A'))} (Separability: higher is better)
- **far_healthy**: {_fmt(metrics.get('far_healthy', 'N/A'))} (False alarm rate: lower is better)
- **arl0_samples**: {_fmt(metrics.get('arl0_samples', 'N/A'))} (Samples between false alarms: higher is better)
- **psi_fault_score**: {_fmt(metrics.get('psi_fault_score', 'N/A'))} (Population stability: lower is better)
- **recent_pct_time_anomalous**: {_fmt(metrics.get('recent_pct_time_anomalous', 'N/A'))}
- **mean_profile_cv_baseline**: {_fmt(metrics.get('mean_profile_cv_baseline', 'N/A'))} (Baseline stability: lower is better)
- **coverage_valid_row_fraction_min**: {_fmt(metrics.get('coverage_valid_row_fraction_min', 'N/A'))}
"""

# --- Main Execution Logic ---

def main(args):
    """
    Main function to run the gearbox config generation and MLflow logging pipeline.
    """
    if mlflow.active_run():
        mlflow.end_run()

    # 1. MLflow Setup
    experiment_name = f"gearbox_fault_monitoring/{args.tenant_id}/{args.machine_id}"
    mlbase = MLflowBase(experiment_name)
    run = mlbase.start_run(run_name=args.run_name)
    run_id = run.info.run_id
    logger.info(f"🚀 Started MLflow Run: {run_id} in Experiment: '{experiment_name}'")

    try:
        # 2. Log Initial Parameters
        params_to_log = {
            "run_name": args.run_name,
            "dataset_filename": args.dataset_filename,
            "tenant_id": args.tenant_id,
            "machine_id": args.machine_id,
            "stoppage_current_threshold": args.stoppage_current_threshold,
            "use_parity_profiles": args.use_parity_profiles,
            "baseline_fraction": args.baseline_fraction,
            "aggregation": args.aggregation,
            "threshold_method": args.threshold_method,
        }
        mlbase.log_params(params_to_log)

        # 3. Load & Prepare Data
        dataset_path = os.path.join(DATA_DIR, args.dataset_filename)
        logger.info(f"💾 Loading data from: {dataset_path}")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please place it in the '{DATA_DIR}' folder.")
        
        df = pd.read_csv(dataset_path)
        drop_cols = [c for c in ["_id", "metaData.tenant_id", "metaData.machine_id"] if c in df.columns]
        if drop_cols: df = df.drop(columns=drop_cols)

        # 4. Feature Engineering: Build Harmonic Profiles
        logger.info("🛠️  Building harmonic profiles...")
        profiles_to_compute = {
            "low_order_current_harmonics": dict(t="c", order_range=(1, 10), parity=None),
            "intermediate_order_current_harmonics": dict(t="c", order_range=(11, 20), parity=None),
            "high_order_current_harmonics": dict(t="c", order_range=(21, 30), parity=None),
        }
        if args.use_parity_profiles:
            profiles_to_compute.update({
                "odd_parity_current_harmonics": dict(t="c", order_range=None, parity="odd"),
                "even_parity_current_harmonics": dict(t="c", order_range=None, parity="even"),
            })

        profile_series_map, profile_valid_fracs = {}, []
        for pname, p_args in profiles_to_compute.items():
            s, dbg = build_profile(df, **p_args)
            profile_series_map[pname] = s
            if dbg["rows"] > 0:
                profile_valid_fracs.append(dbg["valid_rows"] / dbg["rows"])
            if dbg["valid_rows"] < 50:
                 logger.warning(f"{pname} has only {dbg['valid_rows']} valid rows.")
        
        coverage_min = float(np.min(profile_valid_fracs)) if profile_valid_fracs else np.nan
        mlbase.log_metrics({"coverage_valid_row_fraction_min": coverage_min})

        # 5. Build and Log YAML Config
        profile_stats_map = {p: {"mean": v.mean(), "std": v.std(ddof=0)} for p, v in profile_series_map.items()}
        
        config = {"stoppage_current_threshold": args.stoppage_current_threshold, "norm_args": {}}
        for pname, stats in profile_stats_map.items():
            config["norm_args"][pname] = safe_pair(stats["mean"], stats["std"])
            
        output_yaml = f"gearbox_fault_configs_{args.machine_id}.yaml"
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        yaml_path = os.path.join(ARTIFACTS_DIR, output_yaml)
        with open(yaml_path, "w") as f:
            yaml.dump(config, f, sort_keys=False, default_flow_style=False)
        
        mlbase.log_artifact(run_id, local_path=yaml_path)
        logger.info(f"📜 Saved and logged config to MLflow: {output_yaml}")
        
        # 6. Unlabeled Evaluation
        logger.info("📈 Performing unlabeled evaluation...")
        profiles_df = pd.DataFrame(profile_series_map).sort_index()
        split_idx = int(len(profiles_df) * args.baseline_fraction)
        baseline_df, recent_df = profiles_df.iloc[:split_idx], profiles_df.iloc[split_idx:]
        
        baseline_means = baseline_df.mean()
        baseline_stds = baseline_df.std().replace(0, np.nan)
        z_df = ((profiles_df - baseline_means) / baseline_stds).clip(lower=0)
        
        fault_score = z_df.mean(axis=1) if args.aggregation == "mean_z" else z_df.max(axis=1)
        baseline_scores, recent_scores = fault_score.iloc[:split_idx].dropna(), fault_score.iloc[split_idx:].dropna()
        
        thr = np.percentile(baseline_scores, 99) if args.threshold_method == "p99" else baseline_scores.mean() + 3 * baseline_scores.std()
        mlbase.log_params({"eval_decision_threshold": thr})

        # 7. Calculate and Log Core-6 Metrics & Quality Index
        far_healthy = (baseline_scores >= thr).mean()
        metrics_to_log = {
            "drift_cohens_d": cohens_d(baseline_scores, recent_scores),
            "psi_fault_score": psi(baseline_scores, recent_scores),
            "far_healthy": far_healthy,
            "arl0_samples": 1.0 / max(far_healthy, 1e-9),
            "recent_pct_time_anomalous": (recent_scores >= thr).mean(),
            "mean_profile_cv_baseline": (baseline_df.std() / baseline_df.mean()).mean(),
            "eval_decision_threshold": thr
        }
        
        qi_cohen = metrics_to_log.get('drift_cohens_d', 0)
        qi_far = metrics_to_log.get('far_healthy', 0)
        qi_psi = metrics_to_log.get('psi_fault_score', 0)
        qi_cv = metrics_to_log.get('mean_profile_cv_baseline', 0)

        quality_index = (qi_cohen if not np.isnan(qi_cohen) else 0) \
                        - 5.0 * (qi_far if not np.isnan(qi_far) else 0) \
                        + 0.5 * (qi_psi if not np.isnan(qi_psi) else 0) \
                        - 0.5 * (qi_cv if not np.isnan(qi_cv) else 0)

        metrics_to_log["quality_index"] = quality_index
        mlbase.log_metrics({k: v for k,v in metrics_to_log.items() if v is not None and not np.isnan(v)})
        logger.info("📝 Logged final metrics to MLflow.")

        # 8. Set Run Notes
        notes = generate_run_notes(params_to_log, metrics_to_log, output_yaml)
        mlflow.set_tag("mlflow.note.content", notes)

    except Exception as e:
        logger.error(f"❌ An error occurred during the run: {e}", exc_info=True)
        raise
    finally:
        # 9. End MLflow Run
        mlflow.end_run()
        logger.info("🏁 Run finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and log a gearbox fault detection config to MLflow.")

    # example usage:
    # python gearbox_model_training.py --tenant-id "28" --machine-id "257" --dataset-filename "iotts.harmonics_257.csv" --run-name "harmonic_profiling_kstest_v1" --use-parity-profiles
    
    # --- Required Arguments ---
    parser.add_argument("--tenant-id", type=str, required=True, help="Tenant ID for the machine.")
    parser.add_argument("--machine-id", type=str, required=True, help="Machine ID for the new equipment.")
    parser.add_argument("--dataset-filename", type=str, required=True, help=f"Filename of the CSV dataset (must be in the '{DATA_DIR}/' folder).")
    
    # --- Optional Model/Knob Arguments ---
    parser.add_argument("--run-name", type=str, default="harmonic_profiling_v1", help="Name for the approach/run.")
    parser.add_argument("--stoppage-current-threshold", type=float, default=40.0, help="Current threshold to filter non-operating periods.")
    parser.add_argument("--use-parity-profiles", action='store_true', help="Flag to include odd/even parity profiles.")
    parser.add_argument("--baseline-fraction", type=float, default=0.7, help="Portion of data to use as the healthy baseline for evaluation.")
    parser.add_argument("--aggregation", type=str, default="max_z", choices=["max_z", "mean_z"], help="Method to aggregate profile Z-scores into a single fault score.")
    parser.add_argument("--threshold-method", type=str, default="mean+3sigma", choices=["mean+3sigma", "p99"], help="Method to derive the decision threshold from the baseline.")
    
    cli_args = parser.parse_args()
    main(cli_args)
