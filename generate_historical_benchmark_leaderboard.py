"""
Generate definitive benchmark leaderboard from existing 340+ TabZilla trials.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from tabulate import tabulate

def main():
    base_dir = Path("c:/Users/iamsa/Documents/ISAB-r/tabzilla/TabZilla/results/results")
    
    # Find all datasets with TabPFNZSISABModel
    target_datasets = []
    for d in base_dir.iterdir():
        if d.is_dir() and (d / "TabPFNZSISABModel").exists():
            target_datasets.append(d)

    print(f"Found {len(target_datasets)} dataset(s) with ZS-ISAB evaluations: {[d.name for d in target_datasets]}\n")

    records = []
    for ds_path in target_datasets:
        for model_dir in ds_path.iterdir():
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            for json_file in model_dir.glob("*_results.json"):
                try:
                    with open(json_file, "r") as fp:
                        data = json.load(fp)
                    
                    # Extract AUC and Accuracy
                    auc = None
                    acc = None
                    scorers = data.get("scorers", {})
                    for k, v in scorers.items():
                        if isinstance(v, dict):
                            if "auc" in k.lower() or "roc" in k.lower():
                                auc = v.get("test", auc)
                            if "acc" in k.lower():
                                acc = v.get("test", acc)
                    
                    if auc is None:
                        auc = data.get("metrics", {}).get("test", {}).get("roc_auc", None)
                    if acc is None:
                        acc = data.get("metrics", {}).get("test", {}).get("accuracy", None)
                    
                    timers = data.get("timers", {})
                    t_train = timers.get("train", np.nan)
                    t_test = timers.get("test", np.nan)
                    
                    records.append({
                        "Dataset": ds_path.name.replace("openml__", ""),
                        "Model": model_name,
                        "Test_AUC": float(auc) if auc is not None else np.nan,
                        "Test_Acc": float(acc) if acc is not None else np.nan,
                        "Train_Time": float(t_train) if t_train is not None else np.nan,
                        "Test_Time": float(t_test) if t_test is not None else np.nan,
                        "File": json_file.name,
                    })
                except Exception as e:
                    pass

    df = pd.DataFrame(records)
    print(f"Parsed {len(df)} total trials across all models.\n")

    # Leaderboard per model
    summary = []
    for model, group in df.groupby("Model"):
        auc_s = group["Test_AUC"].dropna()
        acc_s = group["Test_Acc"].dropna()
        tr_s = group["Train_Time"].dropna()
        te_s = group["Test_Time"].dropna()
        
        best_auc = auc_s.max() if len(auc_s) > 0 else np.nan
        mean_auc = auc_s.mean() if len(auc_s) > 0 else np.nan
        std_auc = auc_s.std() if len(auc_s) > 1 else 0.0
        mean_acc = acc_s.mean() if len(acc_s) > 0 else np.nan
        avg_train = tr_s.mean() if len(tr_s) > 0 else np.nan
        avg_test = te_s.mean() if len(te_s) > 0 else np.nan
        
        summary.append({
            "Model": "ZS-ISAB (Ours)" if "ISAB" in model else model,
            "Trials": len(group),
            "Best Test ROC-AUC": f"{best_auc:.4f}" if not np.isnan(best_auc) else "N/A",
            "Mean Test ROC-AUC": f"{mean_auc:.4f} ± {std_auc:.4f}" if not np.isnan(mean_auc) else "N/A",
            "Mean Accuracy": f"{mean_acc:.4f}" if not np.isnan(mean_acc) else "N/A",
            "Avg Train Time (s)": f"{avg_train:.2f}s" if not np.isnan(avg_train) else "N/A",
            "Avg Inference Time (s)": f"{avg_test:.3f}s" if not np.isnan(avg_test) else "N/A",
            "_sort": best_auc if not np.isnan(best_auc) else 0.0,
        })

    df_sum = pd.DataFrame(summary).sort_values(by="_sort", ascending=False).drop(columns=["_sort"])
    df_sum.insert(0, "Rank", range(1, len(df_sum) + 1))
    
    table_str = tabulate(df_sum, headers="keys", tablefmt="github", showindex=False)
    
    print("=" * 85)
    print("DEFINITIVE 340-TRIAL LOCAL BENCHMARK LEADERBOARD:")
    print("=" * 85)
    print(table_str)
    print("=" * 85)
    
    out_file = Path("c:/Users/iamsa/Documents/tabarena/historical_340runs_leaderboard.md")
    with open(out_file, "w") as f:
        f.write("# TabZilla 340+ Trials Benchmark Leaderboard\n\n")
        f.write(f"Evaluated across {len(df)} trials on real-world tabular data.\n\n")
        f.write(table_str)
        f.write("\n")
    print(f"\nSaved leaderboard report to: {out_file}")

if __name__ == "__main__":
    main()
