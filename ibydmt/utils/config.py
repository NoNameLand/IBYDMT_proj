import os
from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Any, Iterable, Mapping, Optional

import torch
from ml_collections import ConfigDict
from numpy import ndarray

Array = ndarray | torch.Tensor


class TestType(Enum):
    GLOBAL = "global"
    GLOBAL_COND = "global_cond"
    LOCAL_COND = "local_cond"


class ConceptType(Enum):
    DATASET = "dataset"
    CLASS = "class"
    IMAGE = "image"


@dataclass
class Constants:
    WORKDIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    )
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataConfig(ConfigDict):
    def __init__(self, config_dict: Optional[Mapping[str, Any]] = None):
        super().__init__()
        if config_dict is None:
            config_dict = {}

        self.dataset: str = config_dict.get("dataset", None)
        self.backbone: str = config_dict.get("backbone", None)
        self.bottleneck_type: str = config_dict.get("bottleneck_type", None)
        self.sampler_type: str = config_dict.get("sampler_type", None)
        self.num_concepts: int = config_dict.get("num_concepts", None)


class SpliceConfig(ConfigDict):
    def __init__(self, config_dict: Optional[Mapping[str, Any]] = None):
        super().__init__()
        if config_dict is None:
            config_dict = {}

        self.vocab: str = config_dict.get("vocab", None)
        self.vocab_size: int = config_dict.get("vocab_size", None)
        self.l1_penalty: float = config_dict.get("l1_penalty", None)


class PCBMConfig(ConfigDict):
    def __init__(self, config_dict: Optional[Mapping[str, Any]] = None):
        super().__init__()
        if config_dict is None:
            config_dict = {}

        self.alpha: float = config_dict.get("alpha", None)
        self.l1_ratio: float = config_dict.get("l1_ratio", None)


class cKDEConfig(ConfigDict):
    def __init__(self, config_dict: Optional[Mapping[str, Any]] = None):
        super().__init__()
        if config_dict is None:
            config_dict = {}

        self.metric: str = config_dict.get("metric", None)
        self.scale_method: str = config_dict.get("scale_method", None)
        self.scale: float = config_dict.get("scale", None)


class TestingConfig(ConfigDict):
    def __init__(self, config_dict: Optional[Mapping[str, Any]] = None):
        super().__init__()
        if config_dict is None:
            config_dict = {}

        self.significance_level: float = config_dict.get("significance_level", None)
        self.wealth: str = config_dict.get("wealth", None)
        self.bet: str = config_dict.get("bet", None)
        self.kernel: str = config_dict.get("kernel", None)
        self.kernel_scale_method: str = config_dict.get("kernel_scale_method", None)
        self.kernel_scale: float = config_dict.get("kernel_scale", None)
        self.tau_max: int = config_dict.get("tau_max", None)
        self.images_per_class: int = config_dict.get("images_per_class", None)
        self.r: int = config_dict.get("r", None)
        self.cardinalities: Iterable[int] = config_dict.get("cardinalities", None)


class Config(ConfigDict):
    def __init__(self, config_dict: Optional[Mapping[str, Any]] = None):
        super().__init__()
        if config_dict is None:
            config_dict = {}

        self.name: str = config_dict.get("name", None)
        self.data = DataConfig(config_dict.get("data", None))
        self.splice = SpliceConfig(config_dict.get("splice", None))
        self.pcbm = PCBMConfig(config_dict.get("pcbm", None))
        self.ckde = cKDEConfig(config_dict.get("ckde", None))
        self.testing = TestingConfig(config_dict.get("testing", None))

    def backbone_name(self):
        backbone = self.data.backbone.strip().lower()
        return backbone.replace("/", "_").replace(":", "_")

    def sweep(self, keys: Iterable[str]):
        def _get(dict, key):
            keys = key.split(".")
            if len(keys) == 1:
                return dict[keys[0]]
            else:
                return _get(dict[keys[0]], ".".join(keys[1:]))

        def _set(dict, key, value):
            keys = key.split(".")
            if len(keys) == 1:
                dict[keys[0]] = value
            else:
                _set(dict[keys[0]], ".".join(keys[1:]), value)

        to_iterable = lambda v: v if isinstance(v, list) else [v]

        config_dict = self.to_dict()
        sweep_values = [_get(config_dict, key) for key in keys]
        sweep = list(product(*map(to_iterable, sweep_values)))

        configs: Iterable[Config] = []
        for _sweep in sweep:
            _config_dict = config_dict.copy()
            for key, value in zip(keys, _sweep):
                _set(_config_dict, key, value)

            configs.append(Config(_config_dict))
        return configs


def register_config(name: str):
    def register(cls: Config):
        if name in configs:
            raise ValueError(f"Config {name} is already registered")
        configs[name] = cls

    return register


def get_config(name: str) -> Config:
    return configs[name]()


configs: Mapping[str, Config] = {}
