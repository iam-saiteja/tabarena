"""
Parse all 340+ benchmark trials across all datasets in TabZilla results.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from tabulate import tabulate

def main():
    base_dir = Path("c:/Users/iamsa/Documents/ISAB-r/tabzilla/TabZilla/results/results")
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return
        
    records = []
    
    # Iterate through all dataset directories
    for ds_dir in base_dir.iterdir():
        if not ds_dir.is_dir():
            continue
        ds_name = ds_dir.name.replace("openml__", "")
        
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
                    
                    mean_trial_auc = float(np.mean(test_auc_list)) if test_auc_list else np.nan
                    mean_trial_acc = float(np.mean(test_acc_list)) if test_acc_list else np.nan
                    mean_train_t = float(np.mean(train_t_list)) if train_t_list else np.nan
                    mean_test_t = float(np.mean(test_t_list)) if test_t_list else np.nan
                    
                    clean_model = "ZS-ISAB (Ours)" if "ISAB" in model_name else model_name
                    
                    records.append({
                        "Dataset": ds_name,
                        "Model": clean_model,
                        "Trial": json_file.stem,
                        "Test_AUC": mean_trial_auc,
                        "Test_Acc": mean_trial_acc,
                        "Train_Time_s": mean_train_t,
                        "Test_Time_s": mean_test_t,
                    })
                except Exception:
                    pass

    df = pd.DataFrame(records)
    print(f"Total benchmark trials parsed: {len(df)} across {df['Dataset'].nunique()} datasets and {df['Model'].nunique()} models.\n")

    # Overall Leaderboard
    summary = []
    for model, grp in df.groupby("Model"):
        valid_auc = grp["Test_AUC"].dropna()
        valid_acc = grp["Test_Acc"].dropna()
        valid_tr = grp["Train_Time_s"].dropna()
        valid_te = grp["Test_Time_s"].dropna()
        
        best_auc = valid_auc.max() if len(valid_auc) > 0 else np.nan
        mean_auc = valid_auc.mean() if len(valid_auc) > 0 else np.nan
        std_auc = valid_auc.std() if len(valid_auc) > 1 else 0.0
        mean_acc = valid_acc.mean() if len(valid_acc) > 0 else np.nan
        avg_train = valid_tr.mean() if len(valid_tr) > 0 else np.nan
        avg_test = valid_te.mean() if len(valid_te) > 0 else np.nan
        
        summary.append({
            "Model": model,
            "Trials": len(grp),
            "Best Test ROC-AUC": f"{best_auc:.4f}" if not np.isnan(best_auc) else "N/A",
            "Mean Test ROC-AUC": f"{mean_auc:.4f} ± {std_auc:.4f}" if not np.isnan(mean_auc) else "N/A",
            "Mean Accuracy": f"{mean_acc:.4f}" if not np.isnan(mean_acc) else "N/A",
            "Avg Fit Time (s)": f"{avg_train:.4f}s" if not np.isnan(avg_train) else "N/A",
            "Avg Inference Time (s)": f"{avg_test:.4f}s" if not np.isnan(avg_test) else "N/A",
            "_sort": mean_auc if not np.isnan(mean_auc) else 0.0,
        })

    df_sum = pd.DataFrame(summary).sort_values(by="_sort", ascending=False).drop(columns=["_sort"])
    df_sum.insert(0, "Rank", range(1, len(df_sum) + 1))

    table_str = tabulate(df_sum, headers="keys", tablefmt="github", showindex=False)
    
    print("=" * 95)
    print(f"COMPREHENSIVE MULTI-MODEL BENCHMARK LEADERBOARD ({len(df)} TOTAL TRIALS):")
    print("=" * 95)
    print(table_str)
    print("=" * 95)

    out_file = Path("c:/Users/iamsa/Documents/tabarena/historical_comprehensive_leaderboard.md")
    with open(out_file, "w") as f:
        f.write(f"# Comprehensive TabZilla Benchmark Leaderboard ({len(df)} Total Trials)\n\n")
        f.write(table_str)
        f.write("\n")
    print(f"\nSaved comprehensive leaderboard report to: {out_file}")

if __name__ == "__main__":
    main()
