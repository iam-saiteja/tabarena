from __future__ import annotations

import pandas as pd
import pytest

from tabarena.models.zstabfm import ZSTabFMModel, gen_zstabfm, zstabfm_info, zstabfm_method_metadata


def test_zstabfm_instantiation():
    model = ZSTabFMModel(problem_type="binary")
    assert model.ag_key == "TA-ZSTABFM"
    assert model.ag_name == "TA-ZS-TabFM"
    assert "binary" in model._supported_problem_types


def test_zstabfm_metadata():
    assert zstabfm_info.model_cls == ZSTabFMModel
    assert zstabfm_method_metadata.method == "ZS-TabFM"
    assert zstabfm_method_metadata.ag_key == "TA-ZSTABFM"
    assert gen_zstabfm is not None


def test_zstabfm_memory_estimate():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
    model = ZSTabFMModel()
    mem = model._estimate_memory_usage(df)
    assert isinstance(mem, int) and mem > 0
