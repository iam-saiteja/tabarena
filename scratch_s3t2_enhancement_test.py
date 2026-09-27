import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_breast_cancer, load_wine, load_digits, load_iris, make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
import warnings
warnings.filterwarnings("ignore")

torch.set_num_threads(1)
DEVICE = torch.device("cpu")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Baseline S3T2 Architecture
# ─────────────────────────────────────────────────────────────────────────────
class LinearFeatureTokenizer(nn.Module):
    def __init__(self, n_features, embed_dim=32):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_features, embed_dim) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_features, embed_dim))
        self.cls = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
    def forward(self, x):
        tok = x.unsqueeze(-1) * self.W.unsqueeze(0) + self.b.unsqueeze(0)
        return torch.cat([self.cls.expand(x.size(0), -1, -1), tok], dim=1)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Enhanced Feature Tokenizer: Periodic / Fourier Embeddings
# ─────────────────────────────────────────────────────────────────────────────
class PeriodicFeatureTokenizer(nn.Module):
    """
    Fourier / Periodic Embedding:
    e_d(x) = [sin(2*pi * W_d * x), cos(2*pi * W_d * x)]
    Enables learning sharp non-linear thresholds & steps on tabular features.
    """
    def __init__(self, n_features, embed_dim=32):
        super().__init__()
        half_dim = embed_dim // 2
        self.frequencies = nn.Parameter(torch.randn(n_features, half_dim) * 2.0)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.cls = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
    def forward(self, x):
        # x: (B, D)
        angles = 2.0 * np.pi * x.unsqueeze(-1) * self.frequencies.unsqueeze(0)
        fourier = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        tok = self.proj(fourier)
        return torch.cat([self.cls.expand(x.size(0), -1, -1), tok], dim=1)

class ResBlock(nn.Module):
    def __init__(self, d, drop=0.1):
        super().__init__()
        self.b = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d*2), nn.GELU(), nn.Dropout(drop), nn.Linear(d*2, d), nn.Dropout(drop))
    def forward(self, x): return x + self.b(x)

class S3Net(nn.Module):
    def __init__(self, nf, nc, hidden, depth, edim=32, use_periodic=False):
        super().__init__()
        if use_periodic:
            self.tok = PeriodicFeatureTokenizer(nf, edim)
        else:
            self.tok = LinearFeatureTokenizer(nf, edim)
        tdim = (nf + 1) * edim
        self.proj = nn.Sequential(nn.Linear(tdim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(depth)])
        self.head = nn.Linear(hidden, nc)
        
    def forward(self, x, ml=-1, lam=None, perm=None):
        h = self.proj(self.tok(x).view(x.size(0), -1))
        for i, blk in enumerate(self.blocks):
            h = blk(h)
            if ml == i and lam is not None and perm is not None:
                h = lam * h + (1 - lam) * h[perm]
        return self.head(h)

# ─────────────────────────────────────────────────────────────────────────────
# Training Function for S3T2 (Baseline vs Enhanced)
# ─────────────────────────────────────────────────────────────────────────────
def train_and_eval(X_train, y_train, X_test, y_test, use_periodic=False, use_brier_penalty=False, use_ema=False):
    le = LabelEncoder()
    yt_train = le.fit_transform(y_train)
    classes = le.classes_
    nc = len(classes)
    nf = X_train.shape[1]
    
    qt = QuantileTransformer(output_distribution="normal", random_state=42)
    Xt = torch.tensor(qt.fit_transform(X_train).astype(np.float32))
    yt = torch.tensor(yt_train, dtype=torch.long)
    
    Xte_t = torch.tensor(qt.transform(X_test).astype(np.float32))
    
    cfgs = [(64, 2), (128, 3), (256, 4)]
    nets = nn.ModuleList([S3Net(nf, nc, h, d, edim=32, use_periodic=use_periodic) for h, d in cfgs])
    
    # EMA setup
    if use_ema:
        ema_params = [p.clone().detach() for p in nets.parameters()]
        
    opt = torch.optim.AdamW(nets.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=350)
    
    n = len(Xt)
    nh = max(1, int(n * 0.4))
    yoh = F.one_hot(yt, nc).float()
    unique_classes = yt.unique()
    
    nets.train()
    for _ in range(350):
        opt.zero_grad()
        loss = torch.tensor(0.0)
        for net in nets:
            perm = torch.randperm(n)
            lam = float(np.random.beta(0.3, 0.3))
            ml = np.random.randint(0, max(1, len(net.blocks)))
            ym = lam * yoh + (1 - lam) * yoh[perm]
            lg = net(Xt, ml=ml, lam=lam, perm=perm)
            log_p = F.log_softmax(lg, -1)
            loss_ce = -(ym * log_p).sum(-1).mean()
            
            # Calibration Brier Penalty (RLCD-inspired calibration regularizer)
            if use_brier_penalty:
                probs = F.softmax(lg, -1)
                brier = ((probs - ym) ** 2).sum(-1).mean()
                loss_ce = loss_ce + 0.3 * brier
                
            loss = loss + loss_ce
            
            # Boundary pairs
            if len(unique_classes) > 1:
                hi, hj = [], []
                for _ in range(nh):
                    idx2 = torch.randperm(len(unique_classes))[:2]
                    c1, c2 = unique_classes[idx2[0]], unique_classes[idx2[1]]
                    i1 = (yt == c1).nonzero(as_tuple=True)[0]
                    i2 = (yt == c2).nonzero(as_tuple=True)[0]
                    if len(i1) > 0 and len(i2) > 0:
                        hi.append(i1[torch.randint(len(i1), (1,))].item())
                        hj.append(i2[torch.randint(len(i2), (1,))].item())
                if hi:
                    hi = torch.tensor(hi); hj = torch.tensor(hj)
                    lh = float(np.random.beta(0.5, 0.5))
                    Xh = lh * Xt[hi] + (1 - lh) * Xt[hj]
                    yh = lh * yoh[hi] + (1 - lh) * yoh[hj]
                    loss = loss + 0.5 * -(yh * F.log_softmax(net(Xh), -1)).sum(-1).mean()
                    
        loss.backward()
        nn.utils.clip_grad_norm_(nets.parameters(), 1.0)
        opt.step()
        sched.step()
        
        if use_ema:
            with torch.no_grad():
                for p_ema, p in zip(ema_params, nets.parameters()):
                    p_ema.mul_(0.99).add_(p, alpha=0.01)

    # Load EMA weights if active
    if use_ema:
        with torch.no_grad():
            for p_ema, p in zip(ema_params, nets.parameters()):
                p.copy_(p_ema)
                
    nets.eval()
    with torch.no_grad():
        lg_cal = sum(net(Xt) for net in nets) / len(nets)
        best_T, best_nll = 1.0, 1e9
        for T in np.linspace(0.3, 3.0, 28):
            nll = F.cross_entropy(lg_cal / T, yt).item()
            if nll < best_nll:
                best_nll, best_T = nll, T
                
        lg_test = sum(net(Xte_t) for net in nets) / len(nets)
        probs_test = F.softmax(lg_test / best_T, -1).numpy()
        preds_test = probs_test.argmax(axis=1)
        
    acc = balanced_accuracy_score(le.transform(y_test), preds_test)
    return acc

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark Comparison across 6 Datasets
# ─────────────────────────────────────────────────────────────────────────────
datasets = {
    "breast_cancer": load_breast_cancer(return_X_y=True),
    "wine": load_wine(return_X_y=True),
    "digits": load_digits(return_X_y=True),
    "iris": load_iris(return_X_y=True),
    "synth_hard": make_classification(n_samples=1500, n_features=30, n_informative=15, n_redundant=5, random_state=42),
    "synth_wide": make_classification(n_samples=800, n_features=50, n_informative=20, n_classes=4, n_clusters_per_class=1, random_state=42),
}

print("=" * 80)
print(f"{'Dataset':<16} | {'S3T2 Baseline':<15} | {'S3T2-Pro (Fourier+EMA+Calib)':<25} | {'Gain':<8}")
print("=" * 80)

base_scores, pro_scores = [], []

for dname, (X, y) in datasets.items():
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    b_accs, p_accs = [], []
    for tr, te in skf.split(X, y):
        # Baseline
        acc_base = train_and_eval(X[tr], y[tr], X[te], y[te], use_periodic=False, use_brier_penalty=False, use_ema=False)
        # Enhanced (Fourier Tokenizer + Brier Regularization + EMA weight averaging)
        acc_pro = train_and_eval(X[tr], y[tr], X[te], y[te], use_periodic=True, use_brier_penalty=True, use_ema=True)
        b_accs.append(acc_base)
        p_accs.append(acc_pro)
        
    mean_b = np.mean(b_accs)
    mean_p = np.mean(p_accs)
    base_scores.append(mean_b)
    pro_scores.append(mean_p)
    
    delta = (mean_p - mean_b) * 100
    sign = "+" if delta >= 0 else ""
    print(f"{dname:<16} | {mean_b:<15.4f} | {mean_p:<25.4f} | {sign}{delta:.2f}%")

print("=" * 80)
print(f"{'AVERAGE':<16} | {np.mean(base_scores):<15.4f} | {np.mean(pro_scores):<25.4f} | +{(np.mean(pro_scores) - np.mean(base_scores))*100:.2f}%")
print("=" * 80)
