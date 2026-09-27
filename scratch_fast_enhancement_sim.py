import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_breast_cancer, load_wine, load_digits, load_iris, make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
import warnings
warnings.filterwarnings("ignore")

torch.set_num_threads(4)

# ─────────────────────────────────────────────────────────────────────────────
# Feature Tokenizers
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

class PeriodicFeatureTokenizer(nn.Module):
    def __init__(self, n_features, embed_dim=32):
        super().__init__()
        half_dim = embed_dim // 2
        self.frequencies = nn.Parameter(torch.randn(n_features, half_dim) * 2.0)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.cls = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
    def forward(self, x):
        angles = 2.0 * np.pi * x.unsqueeze(-1) * self.frequencies.unsqueeze(0)
        fourier = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        tok = self.proj(fourier)
        return torch.cat([self.cls.expand(x.size(0), -1, -1), tok], dim=1)

class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.b = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d*2), nn.GELU(), nn.Linear(d*2, d))
    def forward(self, x): return x + self.b(x)

class S3Net(nn.Module):
    def __init__(self, nf, nc, hidden, depth, edim=32, use_periodic=False):
        super().__init__()
        self.tok = PeriodicFeatureTokenizer(nf, edim) if use_periodic else LinearFeatureTokenizer(nf, edim)
        self.proj = nn.Sequential(nn.Linear((nf + 1) * edim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(depth)])
        self.head = nn.Linear(hidden, nc)
    def forward(self, x, ml=-1, lam=None, perm=None):
        h = self.proj(self.tok(x).view(x.size(0), -1))
        for i, blk in enumerate(self.blocks):
            h = blk(h)
            if ml == i and lam is not None and perm is not None:
                h = lam * h + (1 - lam) * h[perm]
        return self.head(h)

def train_eval_fast(X_train, y_train, X_test, y_test, use_periodic=False, use_brier=False, use_ema=False, epochs=120):
    le = LabelEncoder()
    yt_train = le.fit_transform(y_train)
    nc = len(le.classes_)
    nf = X_train.shape[1]

    qt = QuantileTransformer(output_distribution="normal", random_state=42)
    Xt = torch.tensor(qt.fit_transform(X_train).astype(np.float32))
    yt = torch.tensor(yt_train, dtype=torch.long)
    Xte_t = torch.tensor(qt.transform(X_test).astype(np.float32))

    # Fast 2-net ensemble (64, 128)
    nets = nn.ModuleList([
        S3Net(nf, nc, 64, 2, edim=24, use_periodic=use_periodic),
        S3Net(nf, nc, 128, 2, edim=24, use_periodic=use_periodic),
    ])

    if use_ema:
        ema_params = [p.clone().detach() for p in nets.parameters()]

    opt = torch.optim.AdamW(nets.parameters(), lr=5e-3, weight_decay=1e-4)
    n = len(Xt)
    yoh = F.one_hot(yt, nc).float()

    nets.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.tensor(0.0)
        for net in nets:
            perm = torch.randperm(n)
            lam = float(np.random.beta(0.3, 0.3))
            ym = lam * yoh + (1 - lam) * yoh[perm]
            lg = net(Xt, ml=0, lam=lam, perm=perm)
            loss_ce = -(ym * F.log_softmax(lg, -1)).sum(-1).mean()
            if use_brier:
                probs = F.softmax(lg, -1)
                loss_ce = loss_ce + 0.3 * ((probs - ym) ** 2).sum(-1).mean()
            loss = loss + loss_ce

        loss.backward()
        opt.step()

        if use_ema:
            with torch.no_grad():
                for p_ema, p in zip(ema_params, nets.parameters()):
                    p_ema.mul_(0.98).add_(p, alpha=0.02)

    if use_ema:
        with torch.no_grad():
            for p_ema, p in zip(ema_params, nets.parameters()):
                p.copy_(p_ema)

    nets.eval()
    with torch.no_grad():
        lg_test = (nets[0](Xte_t) + nets[1](Xte_t)) / 2.0
        preds = lg_test.argmax(dim=-1).numpy()

    return balanced_accuracy_score(le.transform(y_test), preds)

datasets = {
    "breast_cancer": load_breast_cancer(return_X_y=True),
    "wine": load_wine(return_X_y=True),
    "digits": load_digits(return_X_y=True),
    "iris": load_iris(return_X_y=True),
    "synth_hard": make_classification(n_samples=600, n_features=20, n_informative=10, random_state=42),
    "synth_wide": make_classification(n_samples=500, n_features=30, n_informative=15, n_classes=3, random_state=42),
}

print("=" * 80, flush=True)
print(f"{'Dataset':<16} | {'S3T2 Baseline':<16} | {'S3T2-Pro (Fourier+EMA+Calib)':<28} | {'Delta':<8}", flush=True)
print("=" * 80, flush=True)

base_all, pro_all = [], []

for dname, (X, y) in datasets.items():
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    b_accs, p_accs = [], []
    for tr, te in skf.split(X, y):
        b_accs.append(train_eval_fast(X[tr], y[tr], X[te], y[te], use_periodic=False, use_brier=False, use_ema=False))
        p_accs.append(train_eval_fast(X[tr], y[tr], X[te], y[te], use_periodic=True, use_brier=True, use_ema=True))

    mb = np.mean(b_accs)
    mp = np.mean(p_accs)
    base_all.append(mb)
    pro_all.append(mp)
    delta = (mp - mb) * 100
    sign = "+" if delta >= 0 else ""
    print(f"{dname:<16} | {mb:<16.4f} | {mp:<28.4f} | {sign}{delta:.2f}%", flush=True)

print("=" * 80, flush=True)
print(f"{'OVERALL AVERAGE':<16} | {np.mean(base_all):<16.4f} | {np.mean(pro_all):<28.4f} | +{(np.mean(pro_all) - np.mean(base_all))*100:.2f}%", flush=True)
print("=" * 80, flush=True)
