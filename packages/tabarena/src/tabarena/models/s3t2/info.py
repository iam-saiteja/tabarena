from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.s3t2.hpo import gen_s3t2
from tabarena.models.s3t2.model import S3T2Model

s3t2_method_metadata = MethodMetadata.config(
    method="S3T2",
    suite="tabarena-2026-08-29",
    ag_key="TA-S3T2",
    model_key="S3T2",
    config_default="S3T2_c1_default_BAG_L1",
    can_hpo=True,
    compute="gpu",
    is_bag=False,
    date="2026-09-20",
    date_introduced="2026-09-20",
    reference_url="https://github.com/iam-saiteja/NSA-TabPFN",
    display_name="S3T2",
    verified=True,
    cache_type="local",
)

s3t2_info = ModelInfo(
    model_cls=S3T2Model,
    search_space=gen_s3t2,
    method_metadata=s3t2_method_metadata,
)