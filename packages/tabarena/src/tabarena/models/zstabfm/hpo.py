from __future__ import annotations

from tabarena.models.zstabfm.model import ZSTabFMModel
from tabarena.utils.config_utils import ConfigGenerator

gen_zstabfm = ConfigGenerator(
    model_cls=ZSTabFMModel,
    manual_configs=[
        {"num_prototypes": 1024, "num_draws": 1, "interface": "ensemble"},
    ],
    search_space={
        "num_prototypes": [1024, 2048],
        "interface": ["ensemble", "default"],
    },
)
