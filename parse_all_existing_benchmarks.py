"""
Parses all 340+ existing TabZilla / TabArena benchmark JSONs and generates the definitive Leaderboard.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from tabulate import tabulate

def main():
    results_dir = Path("c:/Users/iamsa/Documents/ISAB-r/tabzilla/TabZilla/results")
    if not results_dir.exists():
        print(f"Directory not found: {results_dir}")
        return
        
    files = list(results_dir.glob("**/*_results.json"))
    print(f"Discovered {len(files)} benchmark trial files across datasets.")
    
    records = []
    for f in files:
        try:
            with open(f, "r") as fp:
                d = json.load(fp)
            
            # Model & Dataset names
            model_name = d.get("model_name", f.parent.name)
            dataset_name = d.get("dataset_name", f.parent.parent.name)
            
            # Scorers
            test_auc = None
            test_acc = None
            scorers = d.get("scorers", {})
            for s_k, s_v in scorers.items():
                if isinstance(s_v, dict):
                    if "auc" in s_k.lower() or "roc" in s_k.lower():
                        test_auc = s_v.get("test", test_auc)
                    if "acc" in s_k.lower():
                        test_acc = s_v.get("test", test_acc)
            
            # Fallback metrics
            metrics = d.get("metrics", {}).get("test", {})
            if test_auc is None:
                test_auc = metrics.get("roc_auc", metrics.get("auc", None))
            if test_acc is None:
                test_acc = metrics.get("acc", metrics.get("accuracy", None))
                
            # Timers
            timers = d.get("timers", {})
            t_train = timers.get("train", np.nan)
            t_test = timers.get("test", np.nan)
            
            records.append({
                "dataset": dataset_name,
                "model": model_name,
                "test_auc": float(test_auc) if test_auc is not None else np.nan,
                "test_acc": float(test_acc) if test_acc is not None else np.nan,
                "train_time_s": float(t_train) if t_train is not None else np.nan,
                "test_time_s": float(t_test) if t_test is not None else np.nan,
                "trial_file": f.name,
            })
        except Exception as e:
            pass
            
    df = pd.DataFrame(records)
    print(f"Successfully processed {len(df)} total model trials across {df['dataset'].nunique()} datasets.\n")
    
    # Generate Leaderboard aggregated across models
    summary = []
    for model, group in df.groupby("model"):
        valid_auc = group["test_auc"].dropna()
        valid_acc = group["test_acc"].dropna()
        valid_train_t = group["train_time_s"].dropna()
        valid_test_t = group["test_time_s"].dropna()
        
        best_auc = valid_auc.max() if len(valid_auc) > 0 else np.nan
        mean_auc = valid_auc.mean() if len(valid_auc) > 0 else np.nan
        mean_acc = valid_acc.mean() if len(valid_acc) > 0 else np.nan
        avg_fit_t = valid_train_t.mean() if len(valid_train_t) > 0 else np.nan
        avg_test_t = valid_test_t.mean() if len(valid_test_t) > 0 else np.nan
        n_trials = len(group)
        
        summary.append({
            "Model": model,
            "Trials": n_trials,
            "Best Test AUC": f"{best_auc:.4f}" if not np.isnan(best_auc) else "N/A",
            "Mean Test AUC": f"{mean_auc:.4f}" if not np.isnan(mean_auc) else "N/A",
            "Mean Accuracy": f"{mean_acc:.4f}" if not np.isnan(mean_acc) else "N/A",
            "Avg Train Time (s)": f"{avg_fit_t:.2f}s" if not np.isnan(avg_fit_t) else "N/A",
            "Avg Inference Time (s)": f"{avg_test_t:.3f}s" if not np.isnan(avg_test_t) else "N/A",
            "_sort_key": mean_auc if not np.isnan(mean_auc) else 0.0,
        })
        
    df_summary = pd.DataFrame(summary).sort_values(by="_sort_key", ascending=False).drop(columns=["_sort_key"])
    
    table_str = tabulate(df_summary, headers="keys", tablefmt="github", showindex=False)
    
    print("=" * 80)
    print(f"DEFINITIVE BENCHMARK LEADERBOARD ({len(df)} TOTAL RUNS):")
    print("=" * 80)
    print(table_str)
    print("=" * 80)
    
    out_file = Path("c:/Users/iamsa/Documents/tabarena/historical_340runs_leaderboard.md")
    with open(out_file, "w") as f:
        f.write("# TabZilla / TabArena 340+ Benchmark Runs Leaderboard\n\n")
        f.write(f"Analyzed {len(df)} trials across {df['dataset'].nunique()} datasets.\n\n")
        f.write(table_str)
        f.write("\n")
    print(f"\nLeaderboard written to {out_file}")

if __name__ == "__main__":
    main()
