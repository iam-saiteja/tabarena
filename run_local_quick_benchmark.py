"""
Fast Local Benchmark Runner for ZS-TabFM & TabArena Models.
Runs a 4-dataset evaluation in ~2 minutes locally or on the server.
"""
from __future__ import annotations

import sys
import time
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from tabulate import tabulate

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "tabarena" / "src"))

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from tabarena.models.zstabfm.model import ZSTabFMModel
    HAS_ZSTABFM = True
except Exception as e:
    HAS_ZSTABFM = False
    ZSTABFM_ERR = str(e)


from sklearn.datasets import load_breast_cancer, load_diabetes, make_regression

def get_benchmark_datasets():
    datasets = []
    
    # 1. Breast Cancer (Binary Classification, 30 features, 569 rows)
    data = load_breast_cancer(as_frame=True)
    datasets.append({
        "name": "Breast Cancer",
        "task": "binary",
        "X": data.data,
        "y": data.target,
    })
    
    # 2. Diabetes Classification (Binarized, 10 features, 442 rows)
    data = load_diabetes(as_frame=True)
    y_bin = (data.target > np.median(data.target)).astype(int)
    datasets.append({
        "name": "Diabetes (Binary)",
        "task": "binary",
        "X": data.data,
        "y": pd.Series(y_bin, name="target"),
    })
    
    # 3. Diabetes Regression (Continuous, 10 features, 442 rows)
    data_reg = load_diabetes(as_frame=True)
    datasets.append({
        "name": "Diabetes (Regression)",
        "task": "regression",
        "X": data_reg.data,
        "y": data_reg.target,
    })

    # 4. Synthetic Tabular Regression (Continuous, 12 features, 600 rows)
    X_syn, y_syn = make_regression(n_samples=600, n_features=12, noise=0.1, random_state=42)
    datasets.append({
        "name": "Synthetic Tabular (600 rows)",
        "task": "regression",
        "X": pd.DataFrame(X_syn, columns=[f"feat_{i}" for i in range(12)]),
        "y": pd.Series(y_syn, name="target"),
    })

    return datasets


def evaluate_model(model_name: str, model_factory, datasets: list[dict]) -> dict:
    scores = []
    train_times = []
    predict_times_per_1k = []
    
    for ds in datasets:
        X, y = ds["X"], ds["y"]
        task = ds["task"]
        
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
        model = model_factory(task)
        
        # Fit timing
        t0 = time.perf_counter()
        if isinstance(model, ZSTabFMModel):
            # AutoGluon AbstractModel style
            model._fit(X=X_tr, y=y_tr, num_cpus=4, num_gpus=0)
        else:
            model.fit(X_tr, y_tr)
        t_fit = time.perf_counter() - t0
        train_times.append(t_fit)
        
        # Inference timing
        t0 = time.perf_counter()
        if task == "binary":
            if hasattr(model, "_predict_proba"):
                y_prob = model._predict_proba(X_te)
                if y_prob.ndim == 2:
                    y_prob = y_prob[:, 1]
            else:
                y_prob = model.predict_proba(X_te)[:, 1]
            t_pred = time.perf_counter() - t0
            score = roc_auc_score(y_te, y_prob)
        else:
            if hasattr(model, "_predict"):
                y_pred = model._predict(X_te)
            else:
                y_pred = model.predict(X_te)
            t_pred = time.perf_counter() - t0
            rmse = np.sqrt(mean_squared_error(y_te, y_pred))
            y_std = np.std(y_te)
            score = max(0.0, 1.0 - (rmse / (y_std + 1e-6)))
        
        p_time_1k = (t_pred / len(X_te)) * 1000.0
        predict_times_per_1k.append(p_time_1k)
        scores.append(score)
        
    return {
        "Model": model_name,
        "Mean Score (AUC / 1-NRMSE)": f"{np.mean(scores):.4f}",
        "Avg Fit Time (s)": f"{np.mean(train_times):.2f}s",
        "Predict Time / 1k (s)": f"{np.mean(predict_times_per_1k):.3f}s",
    }


def main():
    print("=" * 70)
    print("RUNNING RAPID LOCAL TABULAR BENCHMARK (4 DATASETS, ~60 SECONDS)")
    print("=" * 70)
    
    datasets = get_benchmark_datasets()
    print(f"Loaded {len(datasets)} evaluation datasets:")
    for ds in datasets:
        print(f"  - {ds['name']}: {len(ds['X'])} rows, {ds['X'].shape[1]} features ({ds['task']})")
    print("-" * 70)
    
    models = []
    
    def rf_factory(task):
        return RandomForestClassifier(n_estimators=100, random_state=42) if task == "binary" else RandomForestRegressor(n_estimators=100, random_state=42)
    models.append(("RandomForest (default)", rf_factory))
    
    if HAS_LGBM:
        def lgbm_factory(task):
            return LGBMClassifier(n_estimators=100, random_state=42, verbose=-1) if task == "binary" else LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        models.append(("LightGBM (default)", lgbm_factory))

    if HAS_CATBOOST:
        def cb_factory(task):
            return CatBoostClassifier(iterations=150, random_seed=42, verbose=0) if task == "binary" else CatBoostRegressor(iterations=150, random_seed=42, verbose=0)
        models.append(("CatBoost (default)", cb_factory))

    if HAS_ZSTABFM:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        inter = "ensemble" if torch.cuda.is_available() else "default"
        def zstabfm_factory(task):
            return ZSTabFMModel(problem_type=task, hyperparameters={"interface": inter, "num_prototypes": 512, "device": dev})
        models.append((f"ZS-TabFM ({inter.capitalize()})", zstabfm_factory))
    else:
        print(f"Note: ZSTabFM disabled locally: {ZSTABFM_ERR}")
        
    results = []
    for name, factory in models:
        print(f"Evaluating {name}...")
        try:
            res = evaluate_model(name, factory, datasets)
            results.append(res)
        except Exception as e:
            print(f"  Failed {name}: {e}")
            
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="Mean Score (AUC / 1-NRMSE)", ascending=False)
    
    table_str = tabulate(df_results, headers="keys", tablefmt="github", showindex=False)
    print("\n" + "=" * 70)
    print("LOCAL BENCHMARK LEADERBOARD:")
    print("=" * 70)
    print(table_str)
    print("=" * 70)
    
    out_file = Path(__file__).parent / "local_benchmark_leaderboard.md"
    with open(out_file, "w") as f:
        f.write("# Rapid Local Tabular Benchmark Leaderboard\n\n")
        f.write(table_str)
        f.write("\n")
    print(f"\nSaved local leaderboard to {out_file}")


if __name__ == "__main__":
    main()
