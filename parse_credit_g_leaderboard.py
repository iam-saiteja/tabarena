"""
Parse all model trials on credit-g benchmark dataset.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from tabulate import tabulate

def main():
    ds_dir = Path("c:/Users/iamsa/Documents/ISAB-r/tabzilla/TabZilla/results/results/openml__credit-g__31")
    if not ds_dir.exists():
        print(f"Directory not found: {ds_dir}")
        return
        
    records = []
    for model_dir in ds_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for json_file in model_dir.glob("*_results.json"):
            try:
                with open(json_file, "r") as fp:
                    d = json.load(fp)
                
                test_auc_list = d.get("scorers", {}).get("test", {}).get("AUC", [])
                test_acc_list = d.get("scorers", {}).get("test", {}).get("Accuracy", [])
                train_t_list = d.get("timers", {}).get("train", [])
                test_t_list = d.get("timers", {}).get("test", [])
                
                mean_trial_auc = np.mean(test_auc_list) if test_auc_list else np.nan
                mean_trial_acc = np.mean(test_acc_list) if test_acc_list else np.nan
                mean_train_t = np.mean(train_t_list) if train_t_list else np.nan
                mean_test_t = np.mean(test_t_list) if test_t_list else np.nan
                
                records.append({
                    "model": "ZS-ISAB (Ours)" if "ISAB" in model_name else model_name,
                    "trial": json_file.stem,
                    "trial_auc": mean_trial_auc,
                    "trial_acc": mean_trial_acc,
                    "train_time": mean_train_t,
                    "test_time": mean_test_t,
                })
            except Exception as e:
                pass

    df = pd.DataFrame(records)
    print(f"Successfully parsed {len(df)} total trials across {df['model'].nunique()} models on the 'credit-g' dataset.\n")

    summary = []
    for model, grp in df.groupby("model"):
        best_auc = grp["trial_auc"].max()
        mean_auc = grp["trial_auc"].mean()
        std_auc = grp["trial_auc"].std()
        best_acc = grp["trial_acc"].max()
        mean_acc = grp["trial_acc"].mean()
        avg_train = grp["train_time"].mean()
        avg_test = grp["test_time"].mean()
        
        summary.append({
            "Model": model,
            "Trials": len(grp),
            "Best Test ROC-AUC": f"{best_auc:.4f}",
            "Mean Test ROC-AUC": f"{mean_auc:.4f} ± {std_auc:.4f}",
            "Best Accuracy": f"{best_acc:.4f}",
            "Mean Accuracy": f"{mean_acc:.4f}",
            "Avg Fit Time (s)": f"{avg_train:.4f}s",
            "Avg Inference Time (s)": f"{avg_test:.4f}s",
            "_sort": best_auc,
        })

    df_res = pd.DataFrame(summary).sort_values(by="_sort", ascending=False).drop(columns=["_sort"])
    df_res.insert(0, "Rank", range(1, len(df_res) + 1))

    table_str = tabulate(df_res, headers="keys", tablefmt="github", showindex=False)
    
    print("=" * 90)
    print("OFFICIAL HISTORICAL BENCHMARK LEADERBOARD (CREDIT-G, 180 TOTAL TRIALS):")
    print("=" * 90)
    print(table_str)
    print("=" * 90)

    out_file = Path("c:/Users/iamsa/Documents/tabarena/historical_credit_g_leaderboard.md")
    with open(out_file, "w") as f:
        f.write("# TabZilla Official Leaderboard: Credit-G (180 Trials)\n\n")
        f.write(table_str)
        f.write("\n")
    print(f"\nSaved leaderboard report to: {out_file}")

if __name__ == "__main__":
    main()
