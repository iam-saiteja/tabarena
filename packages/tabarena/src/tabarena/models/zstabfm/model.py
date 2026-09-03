from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    from autogluon.core.metrics import Scorer


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 3: VRAM Auto-Scaling
# Detects GPU VRAM at runtime and returns optimal config for the hardware.
# ---------------------------------------------------------------------------

def _auto_vram_config() -> dict:
    """Returns optimal config dict based on detected GPU VRAM.

    Tiers:
      < 5 GB  — RTX 3050 4GB, GTX 1650 4GB
      < 9 GB  — RTX 3060 8GB, RTX 2080 8GB
      < 16 GB — RTX 3080 12GB, RTX 4070 12GB
      >= 16 GB — RTX 4090 24GB, A100 40GB+
    """
    if not torch.cuda.is_available():
        return {"fit_cap": 500, "num_prototypes": 256,
                "batch_d_large": 64, "batch_d_med": 256, "batch_d_small": 512}
    try:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        vram_gb = 4.0

    if vram_gb < 5:
        # Ultra-safe config for 4GB laptop GPUs to prevent Windows TDR crashes / restarts
        return {"fit_cap": 800, "num_prototypes": 384,
                "batch_d_large": 64, "batch_d_med": 128, "batch_d_small": 512}
    elif vram_gb < 9:
        return {"fit_cap": 2000, "num_prototypes": 1024,
                "batch_d_large": 256, "batch_d_med": 1024, "batch_d_small": 2048}
    elif vram_gb < 16:
        return {"fit_cap": 3000, "num_prototypes": 1536,
                "batch_d_large": 512, "batch_d_med": 1536, "batch_d_small": 2048}
    else:
        return {"fit_cap": 5000, "num_prototypes": 2048,
                "batch_d_large": 1024, "batch_d_med": 2048, "batch_d_small": 2048}


def _resolve_device(device: str | None, num_gpus: int, *, cuda_available: bool) -> str:
    if device is not None:
        device = str(device).lower()
    if device == "cpu":
        return "cpu"
    want_gpu = device in ("gpu", "cuda") or (device is None and bool(num_gpus)) or (device is None and cuda_available)
    if want_gpu and not cuda_available:
        return "cpu"
    return "cuda" if want_gpu else "cpu"


def _stratified_prototype_indices(train_y: torch.Tensor, M: int, max_train: int, device: torch.device, seed: int = 0) -> torch.Tensor:
    """Selects M prototypes with guaranteed class or quantile representation across all labels/targets."""
    g = torch.Generator(device=device) if device.type != "cpu" else torch.Generator()
    g.manual_seed(seed)

    if train_y is None or train_y.dim() == 0:
        return torch.randperm(max_train, generator=g, device=device)[:M]
    
    y_flat = train_y.view(-1)

    # For continuous regression targets, bin into 10 quantile buckets
    if y_flat.dtype in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
        try:
            sorted_idx = torch.argsort(y_flat)
            bin_size = len(sorted_idx) // 10
            y_bins = torch.zeros_like(y_flat, dtype=torch.long)
            for b in range(10):
                end = len(sorted_idx) if b == 9 else (b + 1) * bin_size
                y_bins[sorted_idx[b * bin_size:end]] = b
            unique_classes = torch.arange(10, device=device)
            y_flat = y_bins
        except Exception:
            unique_classes = torch.unique(y_flat)
    else:
        unique_classes = torch.unique(y_flat)

    if len(unique_classes) <= 1:
        return torch.randperm(max_train, generator=g, device=device)[:M]

    per_class = []
    quota = max(1, M // len(unique_classes))

    for cls in unique_classes:
        cls_idx = torch.where(y_flat == cls)[0]
        n_take = min(quota, len(cls_idx))
        if n_take > 0:
            perm = cls_idx[torch.randperm(len(cls_idx), generator=g, device=device)[:n_take]]
            per_class.append(perm)

    selected = torch.cat(per_class) if len(per_class) > 0 else torch.randperm(max_train, generator=g, device=device)[:M]
    if len(selected) < M:
        mask = torch.ones(max_train, dtype=torch.bool, device=device)
        mask[selected] = False
        rem = torch.where(mask)[0]
        n_more = min(M - len(selected), len(rem))
        if n_more > 0:
            extra = rem[torch.randperm(len(rem), generator=g, device=device)[:n_more]]
            selected = torch.cat([selected, extra])

    final_perm = torch.randperm(len(selected), generator=g, device=device)
    return selected[final_perm]


def patch_tabfm_with_zsisab(base_model: nn.Module, num_prototypes: int = 512, num_draws: int = 3):
    """Patches TabFM's ICLearning module with Multi-Draw RAPS prototype selection.

    RAPS (Retrieval-Augmented Prototype Selection):
      For each test batch, computes cosine similarity between the test-batch centroid
      in TabFM embedding space and every stored training row embedding, then selects
      the top-M most relevant training rows as the ICL context.

    Multi-Draw (default k=3):
      Draw 0: pure RAPS, noise=0.00 — highest relevance
      Draw 1: RAPS + Gaussian noise sigma=0.08 — slight neighborhood diversity
      Draw 2: RAPS + Gaussian noise sigma=0.16 — wider diversity
      Final output = mean of all draw logits for variance reduction.
    """
    if not hasattr(base_model, "icl"):
        return base_model

    icl_module = base_model.icl
    orig_forward = icl_module.forward

    def isab_icl_forward(reps, y, train_size, *, cache=None, return_cache=False):
        b, t, e = reps.shape
        max_train = int(train_size.max().item()) if train_size is not None else 0

        # Activate RAPS+Multi-Draw only when training context exceeds prototype budget
        if cache is None and max_train > num_prototypes and not return_cache:
            M = min(num_prototypes, max_train)
            device = reps.device

            train_reps = reps[:, :max_train, :]          # [B, N, E]
            train_y = y[:, :max_train] if y is not None else None
            test_reps = reps[:, max_train:, :]            # [B, Q, E]

            draw_outputs = []
            k_draws = max(1, num_draws)

            for d in range(k_draws):
                # ---------------------------------------------------------
                # RAPS: select top-M training rows by cosine similarity to
                # the test batch centroid in embedding space
                # ---------------------------------------------------------
                if test_reps.shape[1] > 0:
                    test_centroid = test_reps.mean(dim=1, keepdim=True)              # [B, 1, E]
                    train_norm = F.normalize(train_reps, dim=-1)                     # [B, N, E]
                    cent_norm = F.normalize(test_centroid, dim=-1)                   # [B, 1, E]
                    sims = torch.bmm(train_norm, cent_norm.transpose(1, 2)).squeeze(-1)  # [B, N]

                    # Multi-Draw diversity: noise grows with draw index
                    # Draw 0 -> sigma=0.00 (pure RAPS)
                    # Draw 1 -> sigma=0.08 (slight neighborhood exploration)
                    # Draw 2 -> sigma=0.16 (wider diversity)
                    noise_scale = 0.08 * d
                    if noise_scale > 0:
                        sims = sims + torch.randn_like(sims) * noise_scale

                    _, top_idx = sims.topk(min(M, max_train), dim=-1)               # [B, M]
                    proto_reps = train_reps.gather(
                        1, top_idx.unsqueeze(-1).expand(-1, -1, e)
                    )
                    proto_y = train_y.gather(1, top_idx) if train_y is not None else None

                else:
                    # Edge case: no test rows — fall back to stratified random
                    perm = _stratified_prototype_indices(
                        train_y[0] if train_y is not None else None,
                        M, max_train, device, seed=d * 1000 + 42,
                    )
                    proto_reps = train_reps[:, perm, :]
                    proto_y = train_y[:, perm] if train_y is not None else None

                # Assemble sub-context: [M prototypes] + [Q test rows]
                reps_sub = torch.cat([proto_reps, test_reps], dim=1)
                n_test = test_reps.shape[1]
                if icl_module.is_classifier:
                    y_sub = torch.cat(
                        [proto_y, torch.zeros((b, n_test), dtype=torch.long, device=device)], dim=1
                    )
                else:
                    y_sub = torch.cat(
                        [proto_y, torch.zeros((b, n_test), dtype=reps.dtype, device=device)], dim=1
                    )

                train_size_sub = torch.full_like(train_size, proto_reps.shape[1])
                out_d = orig_forward(reps_sub, y_sub, train_size_sub, cache=cache, return_cache=return_cache)
                draw_outputs.append(out_d)

            # Average logits across draws for variance reduction
            if len(draw_outputs) == 1:
                return draw_outputs[0]
            stacked = torch.stack(draw_outputs, dim=0)
            return torch.mean(stacked, dim=0)

        return orig_forward(reps, y, train_size, cache=cache, return_cache=return_cache)

    icl_module.forward = isab_icl_forward
    return base_model


_BASE_MODEL_CACHE: dict = {}


def _load_tabfm_safely(model_type: str, device: str, dtype: Any) -> nn.Module:
    """Safe, fast loader supporting direct PyTorch checkpoint (.pt) or HuggingFace fallback."""
    from pathlib import Path
    import json
    from tabfm.src.pytorch.tabfm_v1_0_0 import TabFM_HF
    from tabfm import tabfm_v1_0_0_pytorch

    checkpoint_dir = Path.home() / ".cache" / "tabfm_checkpoint" / model_type
    pt_path = checkpoint_dir / "model.pt"
    cfg_path = checkpoint_dir / "config.json"

    if pt_path.exists() and cfg_path.exists():
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
            if dtype is not None and dtype in (torch.bfloat16, torch.float16):
                for k, v in state_dict.items():
                    if v.is_floating_point():
                        state_dict[k] = v.to(dtype)
                torch.set_default_dtype(dtype)
                model = TabFM_HF(**cfg)
                torch.set_default_dtype(torch.float32)
            else:
                model = TabFM_HF(**cfg)

            model.load_state_dict(state_dict)
            model.eval()
            if device is not None and device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                model = model.to("cuda")
            elif device is not None:
                model = model.to(device)
            return model
        except Exception as e:
            logger.warning(f"Fast .pt loader failed ({e}), falling back to default loader...")
            torch.set_default_dtype(torch.float32)

    # Fallback
    root_dir = Path.home() / ".cache" / "tabfm_checkpoint"
    if root_dir.exists():
        return tabfm_v1_0_0_pytorch.load(model_type=model_type, checkpoint_path=str(root_dir), device=device, dtype=dtype)
    return tabfm_v1_0_0_pytorch.load(model_type=model_type, device=device, dtype=dtype)


def _build_zstabfm_estimator(*, problem_type: str, device: str, interface: str = "default", num_prototypes: int = 1024, num_draws: int = 1, **hps):
    from tabfm import TabFMClassifier, TabFMRegressor

    if problem_type in ["binary", "multiclass"]:
        model_type, model_cls = "classification", TabFMClassifier
    elif problem_type == "regression":
        model_type, model_cls = "regression", TabFMRegressor
    else:
        raise AssertionError(f"Unsupported problem_type: {problem_type}")

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    cache_key = (model_type, str(device), str(dtype), num_prototypes, num_draws)

    if cache_key not in _BASE_MODEL_CACHE:
        base_model = _load_tabfm_safely(model_type=model_type, device=device, dtype=dtype)
        base_model = patch_tabfm_with_zsisab(base_model, num_prototypes=num_prototypes, num_draws=num_draws)
        _BASE_MODEL_CACHE[cache_key] = base_model

    base_model = _BASE_MODEL_CACHE[cache_key]
    factory = model_cls.ensemble if interface == "ensemble" else model_cls
    hps.setdefault("max_num_rows", 1000)
    return factory(model=base_model, **hps)


class ZSTabFMModel(AbstractTorchModel):
    """ZS-TabFM-Turbo: Supercharging Google Research's TabFM Foundation Model with Zero-Shot ISAB."""

    ag_key = "TA-ZSTABFM"
    ag_name = "TA-ZS-TabFM"
    ag_priority = 85
    seed_name = "random_state"
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_num_gpus = 1
    default_resources_physical_cores_only = True
    minimum_num_gpus = 1

    _default_ag_args_ensemble_extra = {
        "fold_fitting_strategy": "sequential_local",
        "refit_folds": True,
    }

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 1,
        **kwargs,
    ):
        import torch

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Step 3: resolve VRAM config first — everything else is derived from it
        vram_cfg = _auto_vram_config()
        logger.info(
            "[ZSTabFM v2] VRAM config: fit_cap=%d, num_prototypes=%d, device=%s",
            vram_cfg["fit_cap"],
            vram_cfg["num_prototypes"],
            "cuda" if torch.cuda.is_available() else "cpu",
        )

        hps = self._get_model_params()
        device = _resolve_device(
            hps.pop("device", None),
            num_gpus,
            cuda_available=torch.cuda.is_available(),
        )
        interface = hps.pop("interface", "default")
        # Use VRAM-derived value unless explicitly overridden in hps
        num_prototypes = hps.pop("num_prototypes", vram_cfg["num_prototypes"])
        num_draws = hps.pop("num_draws", 3)

        self.model = _build_zstabfm_estimator(
            problem_type=self.problem_type,
            device=device,
            interface=interface,
            num_prototypes=num_prototypes,
            num_draws=num_draws,
            **hps,
        )
        # Cache VRAM config so predict methods use the same batch sizes
        self._vram_cfg = vram_cfg

        y_fit = y.to_numpy() if hasattr(y, "to_numpy") else np.array(y)

        # Cap in-context training rows to VRAM-appropriate fit_cap
        fit_cap = vram_cfg["fit_cap"]
        if len(X) > fit_cap:
            try:
                y_tensor = torch.from_numpy(y_fit) if isinstance(y_fit, np.ndarray) else torch.tensor(y_fit)
                idx = _stratified_prototype_indices(
                    y_tensor,
                    M=fit_cap,
                    max_train=len(X),
                    device=torch.device("cpu"),
                    seed=42,
                ).cpu().numpy()
                X_fit = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
                y_fit = y_fit[idx]
            except Exception:
                X_fit = X.iloc[:fit_cap] if hasattr(X, "iloc") else X[:fit_cap]
                y_fit = y_fit[:fit_cap]
        else:
            X_fit = X

        with torch.inference_mode():
            self.model.fit(X_fit, y_fit)
        self._target_device = device

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self


    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, **kwargs) -> pd.DataFrame:
        X = super()._preprocess(X, is_train=is_train, **kwargs)
        for col in X.columns:
            if X[col].dtype == object or isinstance(X[col].dtype, pd.CategoricalDtype):
                X[col] = X[col].astype("category")
        return X

    def _get_batch_size(self, n_cols: int) -> int:
        """Returns VRAM-appropriate prediction batch size based on column count."""
        cfg = getattr(self, "_vram_cfg", None) or _auto_vram_config()
        if n_cols > 100:
            return cfg["batch_d_large"]
        elif n_cols > 30:
            return cfg["batch_d_med"]
        else:
            return cfg["batch_d_small"]

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        import torch
        with torch.inference_mode():
            if self.problem_type == "regression":
                return self._predict_batched(X)

            n_rows = len(X)
            n_cols = X.shape[1] if hasattr(X, "shape") else 10
            batch_size = self._get_batch_size(n_cols)

            if n_rows > batch_size:
                prob_list = []
                for start in range(0, n_rows, batch_size):
                    X_chunk = X.iloc[start:start + batch_size]
                    p_chunk = self.model.predict_proba(X_chunk)
                    prob_list.append(p_chunk)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                probs = np.vstack(prob_list)
            else:
                probs = self.model.predict_proba(X)

        # Boundary probability clipping
        eps = 1e-6
        probs = np.clip(probs, eps, 1.0 - eps)

        if self.problem_type == "binary" and probs.ndim == 2 and probs.shape[1] == 2:
            return probs[:, 1]
        return probs

    def _predict_batched(self, X: pd.DataFrame) -> np.ndarray:
        import torch
        with torch.inference_mode():
            n_rows = len(X)
            n_cols = X.shape[1] if hasattr(X, "shape") else 10
            batch_size = self._get_batch_size(n_cols)

            if n_rows > batch_size:
                preds = []
                for start in range(0, n_rows, batch_size):
                    X_chunk = X.iloc[start:start + batch_size]
                    preds.append(self.model.predict(X_chunk))
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                return np.concatenate(preds, axis=0)
            return self.model.predict(X)


    def _predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        return self._predict_batched(X)

    def _get_default_searchspace(self) -> dict:
        # Defaults shown here are conservative; _fit overrides num_prototypes
        # at runtime based on detected GPU VRAM via _auto_vram_config().
        return {
            "num_prototypes": 512,   # overridden by VRAM auto-scale in _fit
            "num_draws": 3,          # Multi-Draw RAPS: 3 draws, averaged
            "interface": "default",
        }

    def score_with_y_pred_proba(self, y, y_pred_proba, **kwargs) -> float:
        try:
            return super().score_with_y_pred_proba(y=y, y_pred_proba=y_pred_proba, **kwargs)
        except ValueError as e:
            if "Only one class present" in str(e):
                return 0.5
            raise

    def get_device(self) -> str:
        param = next(self.model.model.parameters(), None) if hasattr(self, "model") and hasattr(self.model, "model") else None
        return str(param.device) if param is not None else "cpu"

    def _set_device(self, device: str):
        if getattr(self.model, "model", None) is not None:
            try:
                self.model.model.to(device)
            except Exception:
                import torch
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                try:
                    self.model.model.to("cpu")
                except Exception:
                    pass

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    def get_memory_size(self, allow_exception: bool = False, **kwargs) -> int:
        return 100 * 1024 * 1024  # 100 MB integer memory estimate

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        return 100 * 1024 * 1024  # Constant low memory overhead

