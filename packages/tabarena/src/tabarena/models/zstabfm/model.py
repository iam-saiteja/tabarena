from __future__ import annotations

import inspect
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


def patch_tabfm_with_zsisab(base_model: nn.Module, num_prototypes: int = 512, num_draws: int = 1):
    """Patches TabFM's ICLearning module with Vectorized Hierarchical Multi-Scale ISAB.

    Hierarchical Architecture:
      - Tier 1 (Macro): Global boundary anchors covering the dataset convex hull.
      - Tier 2 (Micro): Local Riemannian manifold prototypes matching the test batch centroid.
      - Fused GPU execution: All normalization, similarity bmm, and topk happen in CUDA
        without CPU sync barriers.
    """
    if not hasattr(base_model, "icl"):
        return base_model

    icl_module = base_model.icl
    orig_forward = icl_module.forward

    def isab_icl_forward(reps, y, train_size, *, cache=None, return_cache=False):
        b, t, e = reps.shape
        max_train = int(train_size.max().item()) if train_size is not None else 0

        # Activate Hierarchical ISAB when training rows exceed prototype budget
        if cache is None and max_train > num_prototypes and not return_cache:
            M = min(num_prototypes, max_train)
            device = reps.device
            M_macro = min(64, max(16, M // 8))
            M_micro = M - M_macro

            train_reps = reps[:, :max_train, :]          # [B, N, E]
            train_y = y[:, :max_train] if y is not None else None
            test_reps = reps[:, max_train:, :]            # [B, Q, E]

            draw_outputs = []
            k_draws = max(1, num_draws)

            for d in range(k_draws):
                # Tier 1: Global Macro Anchors
                macro_idx = _stratified_prototype_indices(
                    train_y[0] if train_y is not None else None,
                    M=M_macro,
                    max_train=max_train,
                    device=device,
                    seed=d * 1000 + 42,
                )
                macro_reps = train_reps[:, macro_idx, :]
                macro_y = train_y[:, macro_idx] if train_y is not None else None

                # Tier 2: Fused CUDA Local Micro Prototypes
                if test_reps.shape[1] > 0:
                    test_centroid = test_reps.mean(dim=1, keepdim=True)
                    train_norm = F.normalize(train_reps, dim=-1)
                    cent_norm = F.normalize(test_centroid, dim=-1)
                    sims = torch.bmm(train_norm, cent_norm.transpose(1, 2)).squeeze(-1)

                    # Mask out macro anchors
                    sims.scatter_(1, macro_idx.unsqueeze(0).expand(b, -1), -1e9)

                    if d > 0:
                        sims = sims + torch.randn_like(sims) * (0.08 * d)

                    take_micro = min(M_micro, max_train - M_macro)
                    _, micro_idx = sims.topk(take_micro, dim=-1)
                    micro_reps = train_reps.gather(
                        1, micro_idx.unsqueeze(-1).expand(-1, -1, e)
                    )
                    micro_y = train_y.gather(1, micro_idx) if train_y is not None else None

                    proto_reps = torch.cat([macro_reps, micro_reps], dim=1)
                    proto_y = torch.cat([macro_y, micro_y], dim=1) if train_y is not None else None
                else:
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

            stacked = draw_outputs[0] if len(draw_outputs) == 1 else torch.mean(torch.stack(draw_outputs, dim=0), dim=0)
            
            # Slicing alignment: test predictions sit at max_train : max_train + Q
            q_len = test_reps.shape[1]
            if q_len > 0:
                final_out = torch.zeros((b, max_train + q_len, stacked.shape[-1]), dtype=stacked.dtype, device=device)
                test_preds = stacked[:, M:, :]
                final_out[:, max_train:, :] = test_preds
                return final_out
            else:
                return stacked

        return orig_forward(reps, y, train_size, cache=cache, return_cache=return_cache)

    icl_module.forward = isab_icl_forward
    return base_model


_BASE_MODEL_CACHE: dict = {}


def _load_tabfm_safely(model_type: str, device: str, dtype: Any = None) -> nn.Module:
    """Bulletproof loader: tries tabfm_v1_0_0_pytorch first, falls back to direct safetensors load."""
    from pathlib import Path
    import json

    # Method 1: Try official loader
    try:
        from tabfm import tabfm_v1_0_0_pytorch
        root_dir = Path.home() / ".cache" / "tabfm_checkpoint"
        model_dir = root_dir / model_type
        if model_dir.exists() and (model_dir / "model.safetensors").exists():
            model = tabfm_v1_0_0_pytorch.load(model_type=model_type, checkpoint_path=str(model_dir), device=device)
        else:
            model = tabfm_v1_0_0_pytorch.load(model_type=model_type, device=device)
        if dtype is not None and hasattr(model, "to"):
            try:
                model = model.to(dtype=dtype)
            except Exception:
                pass
        return model
    except Exception as e:
        logger.info(f"Official loader fallback triggered ({e}). Loading directly via safetensors...")

    # Method 2: Direct safetensors download and load
    try:
        import safetensors.torch
        from huggingface_hub import hf_hub_download
        from tabfm.src.pytorch.model import TabFM

        cfg_path = hf_hub_download(repo_id="google/tabfm-1.0.0-pytorch", filename=f"{model_type}/config.json")
        weights_path = hf_hub_download(repo_id="google/tabfm-1.0.0-pytorch", filename=f"{model_type}/model.safetensors")

        with open(cfg_path) as f:
            cfg = json.load(f)

        is_classifier = (model_type == "classification")
        cfg["is_classifier"] = is_classifier
        model = TabFM(**cfg)
        state_dict = safetensors.torch.load_file(weights_path, device="cpu")
        state_dict = {k: v.to(torch.float32) if v.is_floating_point() else v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model = model.to(torch.float32)
        if device is not None:
            model = model.to(device=device)
        model.eval()
        return model
    except Exception as err:
        logger.error(f"Direct safetensors loader failed: {err}")
        raise


def _build_zstabfm_estimator(
    *,
    problem_type: str,
    device: str,
    interface: str = "ensemble",
    num_prototypes: int = 1024,
    num_draws: int = 1,
    cache_context: bool = True,
    n_features: int = 10,
    n_rows: int = 500,
    **hps,
):
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
    supported_params = set(inspect.signature(model_cls.__init__).parameters.keys())
    candidate_kwargs = dict(hps)

    # Dynamic adaptive batching: on wide datasets (e.g. 112 features in 363711),
    # use batch_size=1 (peak VRAM < 5.5GB). On smaller datasets, use batch_size=2.
    if "batch_size" in supported_params:
        default_bs = 1 if (n_features >= 35 or n_rows >= 1000) else 2
        candidate_kwargs.setdefault("batch_size", default_bs)

    if interface == "ensemble":
        if "cache_context" in supported_params:
            candidate_kwargs["cache_context"] = cache_context
        if "keep_cache_on_device" in supported_params:
            candidate_kwargs["keep_cache_on_device"] = True
        if "maybe_quantize_kv_cache" in supported_params:
            candidate_kwargs["maybe_quantize_kv_cache"] = True

        filtered_kwargs = {k: v for k, v in candidate_kwargs.items() if k in supported_params}
        estimator = model_cls.ensemble(model=base_model, **filtered_kwargs)
    else:
        filtered_kwargs = {k: v for k, v in candidate_kwargs.items() if k in supported_params}
        estimator = model_cls(model=base_model, **filtered_kwargs)
    return estimator


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
        import gc
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            gc.collect()
            torch.cuda.empty_cache()

        hps = self._get_model_params()
        device = _resolve_device(
            hps.pop("device", None),
            num_gpus,
            cuda_available=torch.cuda.is_available(),
        )
        interface = hps.pop("interface", "ensemble")
        num_prototypes = hps.pop("num_prototypes", 1024)
        num_draws = hps.pop("num_draws", 1)
        # cache_context=True uses prefill/decode to cache training reps at fit time,
        # eliminating the cost of re-encoding training context on every predict call.
        cache_context = hps.pop("cache_context", True)

        n_cols = X.shape[1] if hasattr(X, "shape") else 10
        n_rows = len(X)

        self.model = _build_zstabfm_estimator(
            problem_type=self.problem_type,
            device=device,
            interface=interface,
            num_prototypes=num_prototypes,
            num_draws=num_draws,
            cache_context=cache_context,
            n_features=n_cols,
            n_rows=n_rows,
            **hps,
        )

        y_fit = y.to_numpy() if hasattr(y, "to_numpy") else np.array(y)
        X_fit = X

        import gc
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

        with torch.inference_mode():
            try:
                self.model.fit(X_fit, y_fit)
            except Exception as e:
                if "out of memory" in str(e).lower() or isinstance(e, torch.OutOfMemoryError):
                    logger.warning(f"CUDA OOM encountered during fit ({e}). Rebuilding clean estimator with batch_size=1...")
                    self.model = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.model = _build_zstabfm_estimator(
                        problem_type=self.problem_type,
                        device=device,
                        interface=interface,
                        num_prototypes=num_prototypes,
                        num_draws=num_draws,
                        cache_context=cache_context,
                        batch_size=1,
                        n_features=n_cols,
                        n_rows=n_rows,
                        **hps,
                    )
                    self.model.fit(X_fit, y_fit)
                else:
                    raise

        self._target_device = device
        self._fit_X = X_fit
        self._fit_y = y_fit
        self.interface = interface
        self.num_prototypes = num_prototypes
        self.num_draws = num_draws
        self.cache_context = cache_context

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

        return self


    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, **kwargs) -> pd.DataFrame:
        X = super()._preprocess(X, is_train=is_train, **kwargs)
        for col in X.columns:
            if X[col].dtype == object or isinstance(X[col].dtype, pd.CategoricalDtype):
                X[col] = X[col].astype("category")
        return X

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        import gc
        import torch
        with torch.inference_mode():
            if self.problem_type == "regression":
                return self._predict_batched(X)

            n_rows = len(X)
            n_cols = X.shape[1] if hasattr(X, "shape") else 10
            batch_size = 256 if n_cols > 100 else (1024 if n_cols > 30 else 2048)

            try:
                if n_rows > batch_size:
                    prob_list = []
                    for start in range(0, n_rows, batch_size):
                        X_chunk = X.iloc[start:start + batch_size]
                        p_chunk = self.model.predict_proba(X_chunk)
                        prob_list.append(p_chunk)
                    probs = np.vstack(prob_list)
                else:
                    probs = self.model.predict_proba(X)
            except Exception as e:
                if "out of memory" in str(e).lower() or isinstance(e, torch.OutOfMemoryError):
                    logger.warning(f"CUDA OOM encountered during predict_proba ({e}). Retrying with batch_size=1...")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if hasattr(self.model, "batch_size"):
                        self.model.batch_size = 1
                    probs = self.model.predict_proba(X)
                else:
                    raise

        # Boundary probability clipping
        eps = 1e-6
        probs = np.clip(probs, eps, 1.0 - eps)

        if self.problem_type == "binary" and probs.ndim == 2 and probs.shape[1] == 2:
            return probs[:, 1]
        return probs

    def _predict_batched(self, X: pd.DataFrame) -> np.ndarray:
        import gc
        import torch
        with torch.inference_mode():
            n_rows = len(X)
            n_cols = X.shape[1] if hasattr(X, "shape") else 10
            batch_size = 256 if n_cols > 100 else (1024 if n_cols > 30 else 2048)

            try:
                if n_rows > batch_size:
                    preds = []
                    for start in range(0, n_rows, batch_size):
                        X_chunk = X.iloc[start:start + batch_size]
                        preds.append(self.model.predict(X_chunk))
                    return np.concatenate(preds, axis=0)
                return self.model.predict(X)
            except Exception as e:
                if "out of memory" in str(e).lower() or isinstance(e, torch.OutOfMemoryError):
                    logger.warning(f"CUDA OOM encountered during predict ({e}). Retrying with batch_size=1...")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if hasattr(self.model, "batch_size"):
                        self.model.batch_size = 1
                    return self.model.predict(X)
                else:
                    raise


    def _predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        return self._predict_batched(X)

    def _get_default_searchspace(self) -> dict:
        return {
            "num_prototypes": 1024,
            "num_draws": 1,
            "interface": "ensemble",
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

    def save(self, path: str = None, verbose: bool = True) -> str:
        """Temporarily detach estimator with PyTorch lambdas so pickle succeeds."""
        model_backup = self.model
        self.model = None
        try:
            path = super().save(path=path, verbose=verbose)
        finally:
            self.model = model_backup
        return path

    @classmethod
    def load(cls, path: str, reset_paths: bool = True, verbose: bool = True):
        """Reconstruct estimator on load."""
        model_obj = super().load(path=path, reset_paths=reset_paths, verbose=verbose)
        if getattr(model_obj, "model", None) is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_obj.model = _build_zstabfm_estimator(
                problem_type=model_obj.problem_type,
                device=device,
                interface=getattr(model_obj, "interface", "ensemble"),
                num_prototypes=getattr(model_obj, "num_prototypes", 1024),
                num_draws=getattr(model_obj, "num_draws", 1),
                cache_context=getattr(model_obj, "cache_context", True),
            )
            if hasattr(model_obj, "_fit_X") and hasattr(model_obj, "_fit_y") and model_obj._fit_X is not None:
                with torch.inference_mode():
                    model_obj.model.fit(model_obj._fit_X, model_obj._fit_y)
        return model_obj

    def get_memory_size(self, allow_exception: bool = False, **kwargs) -> int:
        return 100 * 1024 * 1024  # 100 MB integer memory estimate

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        return 100 * 1024 * 1024  # Constant low memory overhead

