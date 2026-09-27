import os
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sklearn.datasets import load_breast_cancer, load_wine, load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from typesafe_sdk import TypeSafeClient, Choice

API_KEY = "apikey_215727f40654ed954abf9b594b8bddfa2b42_f301f9a2a5ff2c357e5623c574a3b66185b36e3bce20dc6a5f054bff6b14816e"
os.environ["TYPESAFE_API_KEY"] = API_KEY

client = TypeSafeClient(api_key=API_KEY)

def predict_single_row(args):
    row_dict, target_classes, feature_names = args
    # Format state with clean feature representations
    state = {k: float(f"{v:.3f}") if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else str(v) 
             for k, v in row_dict.items()}
    
    criteria = {f"class_{c}": f"Belongs to category {c}" for c in target_classes}
    question = Choice(
        instructions="Analyze these tabular measurements and predict the most likely target class.",
        criteria=criteria
    )
    
    try:
        resp = client.system_one(state=state, questions={"pred": question})
        choice = resp.answers["pred"].choice
        probs = resp.answers["pred"].probabilities or {}
        # Parse choice back to class
        chosen_class = int(choice.replace("class_", "")) if "class_" in choice else 0
        prob_vec = [probs.get(f"class_{c}", 1.0 / len(target_classes)) for c in target_classes]
        # Normalize
        s = sum(prob_vec) or 1.0
        prob_vec = [p / s for p in prob_vec]
        return chosen_class, prob_vec
    except Exception as e:
        # Fallback uniform
        return target_classes[0], [1.0 / len(target_classes)] * len(target_classes)

def evaluate_jev_on_dataset(name, X, y, sample_limit=30):
    print(f"\nEvaluating Jev on {name} (Test subset: {sample_limit} rows)...")
    df_X = pd.DataFrame(X)
    df_X.columns = [f"feat_{i}" for i in range(df_X.shape[1])]
    
    classes = np.unique(y)
    
    # Take a stratified subset for evaluation to respect API latency
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df_X, y, test_size=min(sample_limit, len(y) // 2), stratify=y, random_state=42)
    
    tasks = [(row.to_dict(), classes, list(df_X.columns)) for _, row in X_test.iterrows()]
    
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(predict_single_row, tasks))
    t1 = time.perf_counter()
    
    preds = [r[0] for r in results]
    probs = np.array([r[1] for r in results])
    
    acc = balanced_accuracy_score(y_test, preds)
    elapsed = t1 - t0
    qps = len(X_test) / elapsed
    
    print(f"[+] {name} Results:")
    print(f"  * Balanced Accuracy: {acc:.4f}")
    print(f"  * Total Time: {elapsed:.2f}s ({qps:.1f} rows/sec with 10 workers)")
    return acc, elapsed

print("=" * 70)
print("TESTING JEV SYSTEM ONE ON 3 TABULAR DATASETS")
print("=" * 70)

datasets = {
    "iris": load_iris(return_X_y=True),
    "wine": load_wine(return_X_y=True),
    "breast_cancer": load_breast_cancer(return_X_y=True),
}

scores = {}
for name, (X, y) in datasets.items():
    acc, elapsed = evaluate_jev_on_dataset(name, X, y, sample_limit=25)
    scores[name] = acc

print("\n" + "=" * 70)
print("FINAL JEV TABULAR EVALUATION SUMMARY")
print("=" * 70)
for k, v in scores.items():
    print(f"{k:<15}: Balanced Accuracy = {v:.4f}")
print(f"Average Accuracy: {np.mean(list(scores.values())):.4f}")
