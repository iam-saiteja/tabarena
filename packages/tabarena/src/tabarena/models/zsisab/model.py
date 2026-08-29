"""Zero-Shot ISAB (ZS-ISAB) model wrapper for AutoGluon / TabArena."""
from __future__ import annotations

import types
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel


def apply_zsisab_shims():
    """Apply compatibility shims for PyTorch Optional and Scikit-Learn force_all_finite."""
    import typing
    import torch.nn.modules.transformer
    torch.nn.modules.transformer.Optional = typing.Optional

    import sklearn.utils.validation as val
    if not hasattr(val, "_zsisab_shimmed"):
        _orig_xy, _orig_arr = val.check_X_y, val.check_array
        val.check_X_y = lambda X, y, **kw: _orig_xy(X, y, **{('ensure_all_finite' if k == 'force_all_finite' else k): v for k, v in kw.items()})
        val.check_array = lambda X, **kw: _orig_arr(X, **{('ensure_all_finite' if k == 'force_all_finite' else k): v for k, v in kw.items()})
        val._zsisab_shimmed = True


def inject_zsisab_to_instance(tabpfn_model, num_prototypes: int = 512, chunk_size: int = 16384):
    """Injects ZS-ISAB forward attention into TabPFN's TransformerEncoderLayer."""
    apply_zsisab_shims()
    from zsisab.wrapper import inject_zsisab
    inject_zsisab(num_prototypes=num_prototypes, chunk_size=chunk_size)



class ZSISABModel(AbstractTorchModel):
    """AutoGluon model wrapper for Zero-Shot ISAB (ZS-ISAB)."""

    ag_key = "ZSISAB"
    ag_name = "ZS-ISAB"
    ag_priority = 105
    seed_name = "random_state"
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_num_gpus = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None
        self._discretizer = None
        self._bin_means = None
        self.model = None

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        return {
            "fold_fitting_strategy": "sequential_local",
            "raise_on_model_failure": False,
        }

    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, y: pd.Series | None = None, **kwargs) -> pd.DataFrame:
        if getattr(self, "_features", None) is not None:
            try:
                X = super()._preprocess(X, **kwargs)
            except Exception:
                pass

        if is_train:
            cat_cols = [c for c in X.columns if str(X[c].dtype) in ["category", "object", "bool", "string"]]
            num_cols = [c for c in X.columns if c not in cat_cols]
            self._cat_cols = cat_cols
            self._num_cols = num_cols

            if len(cat_cols) > 0 and y is not None:
                from sklearn.preprocessing import TargetEncoder
                y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.array(y)
                try:
                    cv_folds = min(5, max(2, len(X) // 2))
                    self._target_encoder = TargetEncoder(smooth="auto", cv=cv_folds, random_state=42)
                    self._target_encoder.fit(X[cat_cols].astype(str), y_arr)
                except Exception:
                    self._target_encoder = None
            else:
                self._target_encoder = None

            if len(cat_cols) > 0:
                from sklearn.preprocessing import OrdinalEncoder
                self._ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                self._ordinal_encoder.fit(X[cat_cols].astype(str))
            else:
                self._ordinal_encoder = None

            if len(num_cols) > 0:
                from sklearn.preprocessing import QuantileTransformer
                n_q = min(1000, max(10, len(X)))
                self._quantile_transformer = QuantileTransformer(n_quantiles=n_q, output_distribution="normal", random_state=42)
                try:
                    num_arr = np.nan_to_num(X[num_cols].to_numpy(dtype=np.float32), nan=0.0)
                    self._quantile_transformer.fit(num_arr)
                except Exception:
                    self._quantile_transformer = None
            else:
                self._quantile_transformer = None

        out_parts = []
        if getattr(self, "_cat_cols", None) and len(self._cat_cols) > 0:
            if getattr(self, "_target_encoder", None) is not None:
                try:
                    cat_trans = self._target_encoder.transform(X[self._cat_cols].astype(str))
                    cat_arr = np.nan_to_num(np.array(cat_trans, dtype=np.float32), nan=0.0)
                    out_parts.append(cat_arr)
                except Exception:
                    cat_trans = self._ordinal_encoder.transform(X[self._cat_cols].astype(str))
                    cat_arr = np.nan_to_num(np.array(cat_trans, dtype=np.float32), nan=0.0)
                    out_parts.append(cat_arr)
            elif getattr(self, "_ordinal_encoder", None) is not None:
                cat_trans = self._ordinal_encoder.transform(X[self._cat_cols].astype(str))
                cat_arr = np.nan_to_num(np.array(cat_trans, dtype=np.float32), nan=0.0)
                out_parts.append(cat_arr)

        if getattr(self, "_num_cols", None) and len(self._num_cols) > 0:
            num_raw = np.nan_to_num(X[self._num_cols].to_numpy(dtype=np.float32), nan=0.0)
            if getattr(self, "_quantile_transformer", None) is not None:
                try:
                    num_trans = self._quantile_transformer.transform(num_raw)
                    out_parts.append(np.nan_to_num(num_trans.astype(np.float32), nan=0.0))
                except Exception:
                    out_parts.append(num_raw)
            else:
                out_parts.append(num_raw)

        if len(out_parts) > 0:
            X_arr = np.hstack(out_parts)
        else:
            X_arr = np.nan_to_num(X.to_numpy(dtype=np.float32), nan=0.0)

        # Enforce max 100 features for TabPFN architecture
        if is_train:
            if X_arr.shape[1] > 100:
                from sklearn.decomposition import TruncatedSVD
                n_comp = min(100, max(1, X_arr.shape[0] - 1), X_arr.shape[1])
                self._dim_reducer = TruncatedSVD(n_components=n_comp, random_state=42)
                X_arr = self._dim_reducer.fit_transform(X_arr)
            else:
                self._dim_reducer = None
        else:
            if getattr(self, "_dim_reducer", None) is not None:
                X_arr = self._dim_reducer.transform(X_arr)
            elif X_arr.shape[1] > 100:
                X_arr = X_arr[:, :100]

        return pd.DataFrame(X_arr, columns=[f"f_{i}" for i in range(X_arr.shape[1])], index=X.index)

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        time_limit: float | None = None,
        **kwargs,
    ) -> None:
        apply_zsisab_shims()

        params = self._get_model_params()
        num_prototypes = params.get("num_prototypes", 512)
        chunk_size = params.get("chunk_size", 16384)
        n_ensemble = params.get("n_ensemble", 8)
        device = "cuda" if (num_gpus is not None and num_gpus > 0) else "cpu"

        from tabpfn import TabPFNClassifier
        self.model = TabPFNClassifier(device=device, N_ensemble_configurations=n_ensemble)

        # Inject ZS-ISAB into this specific instance only
        inject_zsisab_to_instance(self.model, num_prototypes=num_prototypes, chunk_size=chunk_size)

        X_processed = self.preprocess(X, y=y, is_train=True)

        if self.problem_type == "regression":
            from sklearn.preprocessing import KBinsDiscretizer
            y_arr = y.to_numpy(dtype=np.float32) if hasattr(y, "to_numpy") else np.array(y, dtype=np.float32)
            n_bins = min(10, max(2, len(np.unique(y_arr))))
            self._discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
            y_binned = self._discretizer.fit_transform(y_arr.reshape(-1, 1)).ravel().astype(np.int64)
            self._bin_means = np.array(
                [y_arr[y_binned == k].mean() if np.any(y_binned == k) else 0.0 for k in range(n_bins)],
                dtype=np.float32,
            )
            y_fit = y_binned
        else:
            self._discretizer = None
            self._bin_means = None
            y_fit = y.to_numpy() if hasattr(y, "to_numpy") else y

        self.model.fit(X_processed, y_fit, overwrite_warning=True)

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if self.problem_type == "regression":
            return self._predict(X, **kwargs)
        X_processed = self.preprocess(X, is_train=False)
        if hasattr(self.model, "predict_proba"):
            preds = self.model.predict_proba(X_processed)
        else:
            preds = self.model.predict(X_processed)

        if self.problem_type == "binary":
            if isinstance(preds, np.ndarray) and preds.ndim == 2 and preds.shape[1] >= 2:
                return preds[:, 1]
            elif isinstance(preds, np.ndarray) and preds.ndim == 2 and preds.shape[1] == 1:
                return preds.ravel()
            return preds
        return preds

    def _predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        X_processed = self.preprocess(X, is_train=False)
        if self.problem_type == "regression":
            if hasattr(self, "_discretizer") and self._discretizer is not None:
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(X_processed)
                    if self._bin_means is not None:
                        k = min(probs.shape[1], len(self._bin_means))
                        return np.dot(probs[:, :k], self._bin_means[:k]).ravel()
                preds = self.model.predict(X_processed)
                if self._bin_means is not None:
                    return self._bin_means[np.clip(preds.astype(int), 0, len(self._bin_means) - 1)].ravel()
            preds = self.model.predict(X_processed)
            return preds.ravel() if isinstance(preds, np.ndarray) else np.array(preds).ravel()
        return super()._predict(X, **kwargs)

    def score_with_y_pred_proba(self, y, y_pred_proba, metric=None, **kwargs) -> float:
        try:
            return super().score_with_y_pred_proba(y=y, y_pred_proba=y_pred_proba, metric=metric, **kwargs)
        except ValueError:
            return 0.5

    def get_device(self) -> str:
        if self.model is not None and hasattr(self.model, "device"):
            return str(self.model.device)
        return "cpu"

    def _set_device(self, device: str):
        if self.model is not None and hasattr(self.model, "to"):
            self.model.to(device)

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        return int(min(1024 ** 3, X.memory_usage(deep=True).sum() * 10 + 200 * 1024 ** 2))
