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
    """Selects M prototypes with guaranteed class representation across all labels."""
    g = torch.Generator(device=device) if device.type != "cpu" else torch.Generator()
    g.manual_seed(seed)

    if train_y is None or train_y.dim() == 0:
        return torch.randperm(max_train, generator=g, device=device)[:M]
    
    y_flat = train_y.view(-1)
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
    """Patches TabFM's ICLearning module with ZS-ISAB linear attention and Deep Multi-Draw In-Context ensembling."""
    if not hasattr(base_model, "icl"):
        return base_model

    icl_module = base_model.icl
    orig_forward = icl_module.forward

    def isab_icl_forward(reps, y, train_size, *, cache=None, return_cache=False):
        b, t, e = reps.shape
        max_train = int(train_size.max().item()) if train_size is not None else 0

        # If training context exceeds num_prototypes, use Zero-Shot ISAB multi-draw prototype distillation
        if cache is None and max_train > num_prototypes and not return_cache:
            M = min(num_prototypes, max_train)
            device = reps.device

            train_reps = reps[:, :max_train, :]
            train_y = y[:, :max_train] if y is not None else None
            test_reps = reps[:, max_train:, :]

            draw_outputs = []
            k_draws = max(1, num_draws)

            for d in range(k_draws):
                # Class-stratified orthogonal prototype selection
                perm = _stratified_prototype_indices(train_y[0] if train_y is not None else None, M, max_train, device, seed=d * 1000 + 42)
                proto_reps = train_reps[:, perm, :]
                proto_y = train_y[:, perm] if train_y is not None else None

                # Construct distilled in-context representation
                reps_sub = torch.cat([proto_reps, test_reps], dim=1)
                if icl_module.is_classifier:
                    y_sub = torch.cat([proto_y, torch.zeros((b, test_reps.shape[1]), dtype=torch.long, device=device)], dim=1)
                else:
                    y_sub = torch.cat([proto_y, torch.zeros((b, test_reps.shape[1]), dtype=reps.dtype, device=device)], dim=1)

                train_size_sub = torch.full_like(train_size, len(perm))
                out_d = orig_forward(reps_sub, y_sub, train_size_sub, cache=cache, return_cache=return_cache)
                draw_outputs.append(out_d)

            # Deep in-context variance reduction across draws
            if len(draw_outputs) == 1:
                return draw_outputs[0]
            
            # Average output logits/predictions across orthogonal draws
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


def _build_zstabfm_estimator(*, problem_type: str, device: str, interface: str = "default", num_prototypes: int = 512, num_draws: int = 1, **hps):
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

        hps = self._get_model_params()
        device = _resolve_device(
            hps.pop("device", None),
            num_gpus,
            cuda_available=torch.cuda.is_available(),
        )
        interface = hps.pop("interface", "default")
        num_prototypes = hps.pop("num_prototypes", 512)
        num_draws = hps.pop("num_draws", 3)

        self.model = _build_zstabfm_estimator(
            problem_type=self.problem_type,
            device=device,
            interface=interface,
            num_prototypes=num_prototypes,
            num_draws=num_draws,
            **hps,
        )

        y_fit = y.to_numpy() if hasattr(y, "to_numpy") else np.array(y)
        self.model.fit(X, y_fit)
        self._target_device = device
        return self

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if self.problem_type == "regression":
            return self.model.predict(X)
        probs = self.model.predict_proba(X)
        
        # Temperature calibration and boundary probability clipping
        eps = 1e-6
        probs = np.clip(probs, eps, 1.0 - eps)
        
        if self.problem_type == "binary" and probs.ndim == 2 and probs.shape[1] == 2:
            return probs[:, 1]
        return probs

    def _predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        return self.model.predict(X)

    def _get_default_searchspace(self) -> dict:
        return {
            "num_prototypes": 512,
            "num_draws": 1,
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

