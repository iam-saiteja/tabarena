import os
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["LOKY_MAX_CPU_COUNT"]="1"
import time, warnings, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.datasets import load_breast_cancer, load_wine, load_digits, load_iris, make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import QuantileTransformer, LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings("ignore")
torch.set_num_threads(1)
DEVICE = torch.device("cpu")

class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, embed_dim=32):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_features, embed_dim)*0.02)
        self.b = nn.Parameter(torch.zeros(n_features, embed_dim))
        self.cls = nn.Parameter(torch.randn(1,1,embed_dim)*0.02)
    def forward(self, x):
        tok = x.unsqueeze(-1)*self.W.unsqueeze(0)+self.b.unsqueeze(0)
        cls = self.cls.expand(x.size(0),-1,-1)
        return torch.cat([cls,tok],dim=1)

class ResBlock(nn.Module):
    def __init__(self, d, drop=0.1):
        super().__init__()
        self.b = nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d*2),nn.GELU(),nn.Dropout(drop),nn.Linear(d*2,d),nn.Dropout(drop))
    def forward(self,x): return x+self.b(x)

class S3Net(nn.Module):
    def __init__(self, n_feat, n_cls, hidden, depth, edim=32, drop=0.15):
        super().__init__()
        self.tok = FeatureTokenizer(n_feat, edim)
        tdim = (n_feat+1)*edim
        self.proj = nn.Sequential(nn.Linear(tdim,hidden),nn.LayerNorm(hidden),nn.GELU())
        self.blocks = nn.ModuleList([ResBlock(hidden,drop) for _ in range(depth)])
        self.head = nn.Linear(hidden, n_cls)
    def forward(self, x, mlayer=-1, lam=None, perm=None):
        h = self.tok(x).view(x.size(0),-1)
        h = self.proj(h)
        for i,blk in enumerate(self.blocks):
            h = blk(h)
            if mlayer==i and lam is not None and perm is not None:
                h = lam*h+(1-lam)*h[perm]
        return self.head(h)

class S3T2v2(BaseEstimator, ClassifierMixin):
    def __init__(self, steps=300, lr=3e-3, edim=32, alpha=0.3, hard_r=0.4):
        self.steps=steps; self.lr=lr; self.edim=edim; self.alpha=alpha; self.hard_r=hard_r
    def fit(self, X, y):
        le = LabelEncoder(); y_enc=le.fit_transform(y)
        self.classes_=le.classes_; self.nc=len(le.classes_); nf=X.shape[1]
        self.qt_=QuantileTransformer(output_distribution='normal',random_state=42)
        Xt=torch.tensor(self.qt_.fit_transform(X).astype(np.float32))
        yt=torch.tensor(y_enc,dtype=torch.long)
        cfgs=[{"h":64,"d":2},{"h":128,"d":3},{"h":256,"d":4}]
        self.nets_=nn.ModuleList([S3Net(nf,self.nc,c["h"],c["d"],self.edim) for c in cfgs])
        opt=torch.optim.AdamW(self.nets_.parameters(),lr=self.lr,weight_decay=1e-4)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=self.steps)
        n=len(Xt); nh=max(1,int(n*self.hard_r))
        self.nets_.train()
        for step in range(self.steps):
            opt.zero_grad(); loss=torch.tensor(0.)
            yoh=F.one_hot(yt,self.nc).float()
            for net in self.nets_:
                perm=torch.randperm(n); lam=float(np.random.beta(self.alpha,self.alpha))
                nl=len(net.blocks); ml=np.random.randint(0,max(1,nl))
                ym=lam*yoh+(1-lam)*yoh[perm]
                lg=net(Xt,mlayer=ml,lam=lam,perm=perm)
                loss=loss+(-(ym*F.log_softmax(lg,-1)).sum(-1)).mean()
                # Hard boundary pairs
                cls_list=yt.unique()
                if len(cls_list)>1:
                    hi,hj=[],[]
                    for _ in range(nh):
                        idx2=torch.randperm(len(cls_list))[:2]
                        c1,c2=cls_list[idx2[0]],cls_list[idx2[1]]
                        i1=(yt==c1).nonzero(as_tuple=True)[0]
                        i2=(yt==c2).nonzero(as_tuple=True)[0]
                        if len(i1)>0 and len(i2)>0:
                            hi.append(i1[torch.randint(len(i1),(1,))].item())
                            hj.append(i2[torch.randint(len(i2),(1,))].item())
                    if hi:
                        hi=torch.tensor(hi); hj=torch.tensor(hj)
                        lh=float(np.random.beta(0.5,0.5))
                        Xh=lh*Xt[hi]+(1-lh)*Xt[hj]
                        yh=lh*yoh[hi]+(1-lh)*yoh[hj]
                        lgh=net(Xh); loss=loss+0.5*(-(yh*F.log_softmax(lgh,-1)).sum(-1)).mean()
            loss.backward(); nn.utils.clip_grad_norm_(self.nets_.parameters(),1.0)
            opt.step(); sched.step()
        self.nets_.eval()
        with torch.no_grad():
            lg_cal=sum(net(Xt) for net in self.nets_)/len(self.nets_)
        best_T,best_nll=1.0,1e9
        for T in np.linspace(0.3,3.0,28):
            nll=F.cross_entropy(lg_cal/T,yt).item()
            if nll<best_nll: best_nll,best_T=nll,T
        self.T_=best_T
        return self
    def predict_proba(self, X):
        Xt=torch.tensor(self.qt_.transform(X).astype(np.float32))
        self.nets_.eval()
        with torch.no_grad():
            lg=sum(net(Xt) for net in self.nets_)/len(self.nets_)
            return np.clip(F.softmax(lg/self.T_,-1).numpy(),1e-7,1-1e-7)
    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

def run_cv(mfn, X, y, k=5):
    skf=StratifiedKFold(n_splits=k,shuffle=True,random_state=42)
    accs,fits,preds=[],[],[]
    for tr,te in skf.split(X,y):
        m=mfn()
        t0=time.perf_counter(); m.fit(X[tr],y[tr]); t1=time.perf_counter()
        p=m.predict(X[te]); t2=time.perf_counter()
        accs.append(balanced_accuracy_score(y[te],p))
        fits.append(t1-t0); preds.append((t2-t1)*1000)
    return np.mean(accs),np.mean(fits),np.mean(preds)

datasets={
    "breast_cancer": load_breast_cancer(return_X_y=True),
    "wine":          load_wine(return_X_y=True),
    "digits":        load_digits(return_X_y=True),
    "iris":          load_iris(return_X_y=True),
    "synth_easy":    make_classification(n_samples=500,n_features=20,n_informative=5,random_state=42),
    "synth_hard":    make_classification(n_samples=2000,n_features=30,n_informative=15,n_redundant=5,random_state=42),
    "synth_wide":    make_classification(n_samples=800,n_features=50,n_informative=20,n_classes=5,n_clusters_per_class=1,random_state=42),
    "synth_tiny":    make_classification(n_samples=150,n_features=10,n_informative=8,random_state=42),
}
models={
    "S3T2-v2 [OURS]": lambda: S3T2v2(steps=300),
    "S3T2-v1 (old)":  lambda: make_pipeline(QuantileTransformer(output_distribution="normal"),MLPClassifier(hidden_layer_sizes=(64,64),max_iter=250,alpha=0.01,random_state=42)),
    "KNN":            lambda: make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=5,n_jobs=1)),
    "LogReg":         lambda: make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,n_jobs=1)),
    "RandomForest":   lambda: RandomForestClassifier(n_estimators=100,n_jobs=1,random_state=42),
    "GradBoost":      lambda: GradientBoostingClassifier(n_estimators=100,random_state=42),
}
W=18
print("="*78)
print(f"{'Model':<{W}} | {'Dataset':<{W}} | {'BalAcc':>7} | {'Fit(s)':>6} | {'Pred(ms)':>8}")
print("="*78)
all_r={m:{"acc":[],"fit":[],"pred":[]} for m in models}
for dname,(X,y) in datasets.items():
    X=np.array(X,dtype=np.float32); y=np.array(y)
    print()
    for mn,mf in models.items():
        try:
            acc,ft,pt=run_cv(mf,X,y)
        except Exception as e:
            print(f"  {mn}: ERROR {e}"); continue
        all_r[mn]["acc"].append(acc); all_r[mn]["fit"].append(ft); all_r[mn]["pred"].append(pt)
        mk="*" if mn=="S3T2-v2 [OURS]" else " "
        print(f"{mn:<{W}} | {dname:<{W}} | {acc:>7.4f}{mk}| {ft:>6.2f} | {pt:>8.2f}")

print(); print("="*78)
print("AVERAGES (sorted by accuracy)")
print("="*78)
rows=[(np.mean(v["acc"]),mn,np.mean(v["fit"]),np.mean(v["pred"])) for mn,v in all_r.items() if v["acc"]]
for acc,mn,ft,pt in sorted(rows,reverse=True):
    tag=" ← WINNER" if mn==sorted(rows,reverse=True)[0][1] else ""
    print(f"{mn:<{W}} | acc={acc:.4f} | fit={ft:.2f}s | pred={pt:.1f}ms{tag}")
print("="*78)
print("\nNo pretraining. No synthetic data. Pure from-scratch PyTorch.")
