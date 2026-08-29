from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.zstabfm.hpo import gen_zstabfm
from tabarena.models.zstabfm.model import ZSTabFMModel

zstabfm_method_metadata = MethodMetadata.config(
    method="ZS-TabFM",
    suite="tabarena-2026-08-29",
    ag_key="TA-ZSTABFM",
    model_key="ZSTABFM",
    config_default="ZS-TabFM_c1_default_BAG_L1",
    can_hpo=True,
    compute="gpu",
    is_bag=False,
    date="2026-08-29",
    date_introduced="2026-08-29",
    reference_url="https://github.com/iam-saiteja/NSA-TabPFN",
    display_name="ZS-TabFM",
    verified=True,
    cache_type="local",
)

zstabfm_info = ModelInfo(
    model_cls=ZSTabFMModel,
    search_space=gen_zstabfm,
    method_metadata=zstabfm_method_metadata,
)
