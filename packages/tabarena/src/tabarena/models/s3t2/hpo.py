from __future__ import annotations

from tabarena.models.s3t2.model import S3T2Model
from tabarena.utils.config_utils import ConfigGenerator

gen_s3t2 = ConfigGenerator(
    model_cls=S3T2Model,
    manual_configs=[
        {
            "steps": 400,
            "lr": 3e-3,
            "embed_dim": 32,
            "mixup_alpha": 0.3,
            "hard_ratio": 0.4,
        },
    ],
    search_space={
        "steps": [300, 500],
        "lr": [1e-3, 3e-3, 5e-3],
        "embed_dim": [16, 32, 64],
        "mixup_alpha": [0.2, 0.3, 0.4],
        "hard_ratio": [0.2, 0.4, 0.6],
    },
)