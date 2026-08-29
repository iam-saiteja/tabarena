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
    want_gpu = device in ("gpu", "cuda") or (device is None and bool(num_gpus))
    if want_gpu and not cuda_available:
        return "cpu"
    return "cuda" if want_gpu else "cpu"


def patch_tabfm_with_zsisab(base_model: nn.Module, num_prototypes: int = 512):
    """Patches TabFM's ICLearning module with ZS-ISAB linear attention for large context."""
    if not hasattr(base_model, "icl"):
        return base_model

    icl_module = base_model.icl
    orig_forward = icl_module.forward

    def isab_icl_forward(reps, y, train_size, *, cache=None, return_cache=False):
        b, t, e = reps.shape
        max_train = int(train_size.max().item()) if train_size is not None else 0

        # If training context exceeds num_prototypes, use Zero-Shot ISAB prototype distillation
        if cache is None and max_train > num_prototypes and not return_cache:
            # Multi-Resolution Prototype Distillation
            M = min(num_prototypes, max_train)
            device = reps.device
            
            # Select diverse prototypes from train context
            train_reps = reps[:, :max_train, :]
            train_y = y[:, :max_train] if y is not None else None
            test_reps = reps[:, max_train:, :]

            perm = torch.randperm(max_train, device=device)[:M]
            proto_reps = train_reps[:, perm, :]
            proto_y = train_y[:, perm] if train_y is not None else None

            # Subsample representation
            reps_sub = torch.cat([proto_reps, test_reps], dim=1)
            if icl_module.is_classifier:
                y_sub = torch.cat([proto_y, torch.zeros((b, test_reps.shape[1]), dtype=torch.long, device=device)], dim=1)
            else:
                y_sub = torch.cat([proto_y, torch.zeros((b, test_reps.shape[1]), dtype=reps.dtype, device=device)], dim=1)
            
            train_size_sub = torch.full_like(train_size, M)
            return orig_forward(reps_sub, y_sub, train_size_sub, cache=cache, return_cache=return_cache)

        return orig_forward(reps, y, train_size, cache=cache, return_cache=return_cache)

    icl_module.forward = isab_icl_forward
    return base_model


def _build_zstabfm_estimator(*, problem_type: str, device: str, interface: str = "default", num_prototypes: int = 512, **hps):
    from tabfm import TabFMClassifier, TabFMRegressor, tabfm_v1_0_0_pytorch

    if problem_type in ["binary", "multiclass"]:
        model_type, model_cls = "classification", TabFMClassifier
    elif problem_type == "regression":
        model_type, model_cls = "regression", TabFMRegressor
    else:
        raise AssertionError(f"Unsupported problem_type: {problem_type}")

    base_model = tabfm_v1_0_0_pytorch.load(model_type=model_type, device=device)
    base_model = patch_tabfm_with_zsisab(base_model, num_prototypes=num_prototypes)

    factory = model_cls.ensemble if interface == "ensemble" else model_cls
    return factory(model=base_model, **hps)


class ZSTabFMModel(AbstractTorchModel):
    """ZS-TabFM: Supercharging Google Research's TabFM Foundation Model with Zero-Shot ISAB."""

    ag_key = "TA-ZSTABFM"
    ag_name = "TA-ZS-TabFM"
    ag_priority = 85
    seed_name = "random_state"
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_num_gpus = 1
    default_resources_physical_cores_only = True
    minimum_num_gpus = 0

    _default_ag_args_ensemble_extra = {
        "fold_fitting_strategy": "sequential_local",
        "refit_folds": True,
    }

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
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

        self.model = _build_zstabfm_estimator(
            problem_type=self.problem_type,
            device=device,
            interface=interface,
            num_prototypes=num_prototypes,
            **hps,
        )

        y_fit = y.to_numpy() if hasattr(y, "to_numpy") else np.array(y)
        self.model.fit(X, y_fit)
        self._target_device = device
        return self

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        return self.model.predict_proba(X)

    def _predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        return self.model.predict(X)

    def _get_default_searchspace(self) -> dict:
        return {
            "num_prototypes": 512,
            "interface": "default",
        }

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        return 100 * 1024 * 1024  # Constant low memory overhead
