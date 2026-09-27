"""
Local simulation: Zero-Training Kernel Meta-Estimator (ZT-KME)
Tests the new architecture against real sklearn datasets.
No pretraining. No synthetic data generation. Pure geometric inference.
"""
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, fetch_openml, load_digits
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Core: Zero-Training Kernel Meta-Estimator
# ─────────────────────────────────────────────────────────────────────────────

class ZTKernelMetaEstimator(BaseEstimator, ClassifierMixin):
    """
    Zero-Training Kernel Meta-Estimator (ZT-KME).
    
    No gradient descent. No pretraining. No synthetic data.
    
    The idea:
    1. Compute per-feature importance from training variance ratio (free, O(ND))
    2. Build a Nystrom-approximated kernel matrix from M landmark prototypes
    3. Use Nadaraya-Watson weighted voting for classification
    4. Ensemble 3 kernel families: RBF + Laplacian + Polynomial (temperature tuned per dataset)
    """
    
    def __init__(self, n_prototypes=256, kernel_temp=1.0, mix_alpha=0.5):
        self.n_prototypes = n_prototypes
        self.kernel_temp = kernel_temp
        self.mix_alpha = mix_alpha
    
    def _compute_feature_weights(self, X, y):
        """Differential Feature Scoring: weight = between-class variance / within-class variance."""
        classes = np.unique(y)
        grand_mean = X.mean(axis=0)
        
        # Between-class variance (weighted by class size)
        between = np.zeros(X.shape[1])
        within = np.zeros(X.shape[1])
        
        for c in classes:
            mask = y == c
            Xc = X[mask]
            nc = mask.sum()
            class_mean = Xc.mean(axis=0)
            between += nc * (class_mean - grand_mean) ** 2
            within += ((Xc - class_mean) ** 2).sum(axis=0)
        
        # Fisher discriminant ratio per feature
        eps = 1e-8
        fisher = between / (within + eps)
        
        # Normalize to [0.1, 2.0] range so weak features still contribute
        w = (fisher - fisher.min()) / (fisher.max() - fisher.min() + eps)
        return 0.1 + 1.9 * w
    
    def _select_prototypes(self, X, y):
        """Stratified landmark selection across class boundaries."""
        classes, counts = np.unique(y, return_counts=True)
        protos = []
        quota = max(1, self.n_prototypes // len(classes))
        
        for c in classes:
            idx = np.where(y == c)[0]
            take = min(quota, len(idx))
            chosen = idx[np.random.choice(len(idx), take, replace=False)]
            protos.append(chosen)
        
        proto_idx = np.concatenate(protos)
        
        # Fill remaining budget
        remaining = self.n_prototypes - len(proto_idx)
        if remaining > 0:
            all_idx = np.arange(len(X))
            pool = np.setdiff1d(all_idx, proto_idx)
            if len(pool) > 0:
                extra = pool[np.random.choice(len(pool), min(remaining, len(pool)), replace=False)]
                proto_idx = np.concatenate([proto_idx, extra])
        
        return proto_idx
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Feature importance weights
        self.feature_weights_ = self._compute_feature_weights(X, y)
        
        # Weighted training data
        Xw = X * self.feature_weights_
        
        # Select prototypes
        np.random.seed(42)
        proto_idx = self._select_prototypes(Xw, y)
        self.X_proto_ = Xw[proto_idx]
        self.y_proto_ = y[proto_idx]
        
        # Tune bandwidth per dataset: median heuristic on prototypes
        dists = cdist(self.X_proto_[:min(100, len(self.X_proto_))], 
                     self.X_proto_[:min(100, len(self.X_proto_))], 'euclidean')
        self.bandwidth_ = np.median(dists[dists > 0]) * self.kernel_temp + 1e-8
        
        return self
    
    def _rbf_kernel(self, D):
        """Gaussian / RBF kernel."""
        return np.exp(-(D**2) / (2 * self.bandwidth_**2))
    
    def _laplacian_kernel(self, D):
        """Laplacian kernel - heavier tails than RBF, better for outliers."""
        return np.exp(-D / self.bandwidth_)
    
    def _polynomial_kernel(self, X_test, X_proto):
        """Polynomial kernel (degree 3) - captures non-linear feature interactions."""
        dot = X_test @ X_proto.T
        return ((dot / (self.bandwidth_ + 1e-8)) + 1) ** 3
    
    def predict_proba(self, X):
        Xw = X * self.feature_weights_
        
        # Compute distances to prototypes
        D = cdist(Xw, self.X_proto_, 'euclidean')
        
        # Ensemble of 3 kernel families
        K_rbf = self._rbf_kernel(D)
        K_lap = self._laplacian_kernel(D)
        K_poly = self._polynomial_kernel(Xw, self.X_proto_)
        K_poly = np.clip(K_poly, 0, None)
        # Normalize polynomial kernel
        row_max = K_poly.max(axis=1, keepdims=True) + 1e-8
        K_poly = K_poly / row_max
        
        # Temperature-scaled ensemble: blend kernels
        a = self.mix_alpha
        K = a * K_rbf + (1 - a) * 0.5 * K_lap + (1 - a) * 0.5 * K_poly
        
        # Nadaraya-Watson: weighted vote per class
        probs = np.zeros((len(X), self.n_classes_))
        for i, cls in enumerate(self.classes_):
            mask = self.y_proto_ == cls
            probs[:, i] = K[:, mask].sum(axis=1)
        
        # Normalize to probabilities
        row_sums = probs.sum(axis=1, keepdims=True) + 1e-12
        probs = probs / row_sums
        
        return np.clip(probs, 1e-8, 1 - 1e-8)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark against KNN, LogReg, and RandomForest baselines
# ─────────────────────────────────────────────────────────────────────────────

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

DATASETS = {
    "breast_cancer": load_breast_cancer(return_X_y=True),
    "wine":          load_wine(return_X_y=True),
    "iris":          load_iris(return_X_y=True),
    "digits":        load_digits(return_X_y=True),
}

MODELS = {
    "ZT-KME (Ours)":     Pipeline([("scaler", StandardScaler()), ("clf", ZTKernelMetaEstimator(n_prototypes=128, kernel_temp=0.7, mix_alpha=0.5))]),
    "KNN (k=5)":         Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
    "LogReg":            Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]),
    "RandomForest":      RandomForestClassifier(n_estimators=100, random_state=42),
    "GradBoost":         GradientBoostingClassifier(n_estimators=100, random_state=42),
}

print("=" * 90)
print(f"{'Dataset':<20} {'ZT-KME':>12} {'KNN':>12} {'LogReg':>12} {'RF':>12} {'GBT':>12}")
print("=" * 90)

results = {m: [] for m in MODELS}

for dname, (X, y) in DATASETS.items():
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    row = {}
    for mname, model in MODELS.items():
        scores = cross_val_score(model, X, y, cv=skf, scoring="balanced_accuracy", n_jobs=1)
        row[mname] = scores.mean()
        results[mname].append(scores.mean())
    
    print(f"{dname:<20} ", end="")
    for mname in MODELS:
        mark = " ★" if mname == "ZT-KME (Ours)" and row[mname] == max(row.values()) else "  "
        print(f"{row[mname]:>10.4f}{mark}", end="")
    print()

print("=" * 90)
print(f"{'AVERAGE':<20} ", end="")
for mname in MODELS:
    avg = np.mean(results[mname])
    print(f"{avg:>12.4f}", end="")
print()
print("=" * 90)
print("\nDone. No pretraining. No synthetic data. Pure geometry.")
