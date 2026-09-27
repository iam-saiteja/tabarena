import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import time
import numpy as np
import warnings
from sklearn.datasets import load_breast_cancer, load_wine, load_digits, load_iris, make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.pipeline import make_pipeline
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class SparkV1(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = make_pipeline(
            StandardScaler(),
            Nystroem(kernel='rbf', n_components=200, random_state=42),
            CalibratedClassifierCV(LogisticRegression(max_iter=1000, n_jobs=1), method='sigmoid', cv=2)
        )
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    def predict(self, X):
        return self.model.predict(X)

class SparkV2(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    def fit(self, X, y):
        X = np.array(X)
        self.scaler = StandardScaler()
        X_s = self.scaler.fit_transform(X)
        
        self.kbest = SelectKBest(f_classif, k=min(10, X_s.shape[1]))
        X_k = self.kbest.fit_transform(X_s, y)
        
        cross_terms = []
        for i in range(X_k.shape[1]):
            for j in range(i, X_k.shape[1]):
                cross_terms.append(X_k[:, i] * X_k[:, j])
        if cross_terms:
            X_cross = np.column_stack(cross_terms)
            X_s = np.hstack([X_s, X_cross])
            
        self.nys_rbf = Nystroem(kernel='rbf', n_components=100, random_state=42)
        self.nys_poly = Nystroem(kernel='poly', degree=2, n_components=100, random_state=42)
        
        X_rbf = self.nys_rbf.fit_transform(X_s)
        X_poly = self.nys_poly.fit_transform(X_s)
        X_features = np.hstack([X_rbf, X_poly])
        
        self.clf = CalibratedClassifierCV(LogisticRegression(max_iter=1000, n_jobs=1), method='isotonic', cv=2)
        try:
            self.clf.fit(X_features, y)
        except Exception:
            # Fallback if isotonic fails for multiclass or small samples
            self.clf = CalibratedClassifierCV(LogisticRegression(max_iter=1000, n_jobs=1), method='sigmoid', cv=2)
            self.clf.fit(X_features, y)
        return self

    def predict(self, X):
        X_s = self.scaler.transform(X)
        X_k = self.kbest.transform(X_s)
        cross_terms = []
        for i in range(X_k.shape[1]):
            for j in range(i, X_k.shape[1]):
                cross_terms.append(X_k[:, i] * X_k[:, j])
        if cross_terms:
            X_cross = np.column_stack(cross_terms)
            X_s = np.hstack([X_s, X_cross])
            
        X_rbf = self.nys_rbf.transform(X_s)
        X_poly = self.nys_poly.transform(X_s)
        X_features = np.hstack([X_rbf, X_poly])
        return self.clf.predict(X_features)

class DART(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    def fit(self, X, y):
        self.scaler = StandardScaler()
        X_s = self.scaler.fit_transform(X)
        
        n_clusters = min(16, len(X_s) // 5)
        n_clusters = max(2, n_clusters)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=1)
        self.kmeans.fit(X_s)
        
        distances = self.kmeans.transform(X_s)
        gamma = 1.0 / (X_s.shape[1] + 1)
        weights = np.exp(-gamma * distances**2)
        weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)
        
        experts_features = []
        for i in range(n_clusters):
            experts_features.append(X_s * weights[:, i:i+1])
        X_expanded = np.hstack(experts_features)
        
        self.clf = RidgeClassifier()
        self.clf.fit(X_expanded, y)
        return self
        
    def predict(self, X):
        X_s = self.scaler.transform(X)
        distances = self.kmeans.transform(X_s)
        gamma = 1.0 / (X_s.shape[1] + 1)
        weights = np.exp(-gamma * distances**2)
        weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)
        
        experts_features = []
        for i in range(self.kmeans.n_clusters):
            experts_features.append(X_s * weights[:, i:i+1])
        X_expanded = np.hstack(experts_features)
        
        return self.clf.predict(X_expanded)

from sklearn.neural_network import MLPClassifier
class S3T2(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.scaler = QuantileTransformer()
        self.mlp = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=250, alpha=0.01, random_state=42)
        
    def fit(self, X, y):
        X_s = self.scaler.fit_transform(X)
        
        X_aug, y_aug = [], []
        classes = np.unique(y)
        for c in classes:
            idx = np.where(y == c)[0]
            if len(idx) > 1:
                for _ in range(len(idx)):
                    i, j = np.random.choice(idx, 2, replace=False)
                    lam = np.random.beta(0.2, 0.2)
                    X_aug.append(lam * X_s[i] + (1 - lam) * X_s[j])
                    y_aug.append(c)
        if X_aug:
            X_s = np.vstack([X_s, np.array(X_aug)])
            y_aug_arr = np.array(y_aug)
            y_full = np.hstack([y, y_aug_arr])
        else:
            y_full = y
            
        self.mlp.fit(X_s, y_full)
        return self
        
    def predict(self, X):
        X_s = self.scaler.transform(X)
        return self.mlp.predict(X_s)

def run_simulation():
    datasets = {
        'breast_cancer': load_breast_cancer(return_X_y=True),
        'wine': load_wine(return_X_y=True),
        'digits': load_digits(return_X_y=True),
        'iris': load_iris(return_X_y=True),
        'make_classif_5': make_classification(n_informative=5, n_samples=500, random_state=42),
        'make_classif_15': make_classification(n_informative=15, n_samples=2000, random_state=42)
    }
    
    models = {
        'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, n_jobs=1)),
        'LogisticReg': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, n_jobs=1)),
        'RandomForest': RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42),
        'GradBoost': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SPARK-v1': SparkV1(),
        'SPARK-v2': SparkV2(),
        'DART': DART(),
        'S3T2': S3T2()
    }
    
    print(f"{'Model':<15} | {'Dataset':<15} | {'Acc':<6} | {'Fit (s)':<7} | {'Pred (ms)':<9}")
    print("-" * 65)
    
    results = {m: {'acc': [], 'fit': [], 'pred': []} for m in models}
    
    for d_name, (X, y) in datasets.items():
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for m_name, model in models.items():
            acc_list, fit_list, pred_list = [], [], []
            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                t0 = time.perf_counter()
                model.fit(X_train, y_train)
                t1 = time.perf_counter()
                
                preds = model.predict(X_test)
                t2 = time.perf_counter()
                
                acc_list.append(balanced_accuracy_score(y_test, preds))
                fit_list.append(t1 - t0)
                pred_list.append((t2 - t1) * 1000.0)
                
            mean_acc = np.mean(acc_list)
            mean_fit = np.mean(fit_list)
            mean_pred = np.mean(pred_list)
            
            results[m_name]['acc'].append(mean_acc)
            results[m_name]['fit'].append(mean_fit)
            results[m_name]['pred'].append(mean_pred)
            
            print(f"{m_name:<15} | {d_name:<15} | {mean_acc:.4f} | {mean_fit:.4f}  | {mean_pred:.4f}")
            
    print("=" * 65)
    print("AVERAGE ACROSS DATASETS")
    print("-" * 65)
    for m_name in models:
        m_acc = np.mean(results[m_name]['acc'])
        m_fit = np.mean(results[m_name]['fit'])
        m_pred = np.mean(results[m_name]['pred'])
        print(f"{m_name:<15} | {'Average':<15} | {m_acc:.4f} | {m_fit:.4f}  | {m_pred:.4f}")

if __name__ == "__main__":
    run_simulation()
