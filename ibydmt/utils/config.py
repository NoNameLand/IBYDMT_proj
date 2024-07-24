import os
from enum import Enum

import torch
from ml_collections import ConfigDict, FrozenConfigDict
from numpy import ndarray

Array = ndarray | torch.Tensor


configs = {}


def register_config(name):
    def register(cls):
        if name in configs:
            raise ValueError(f"Config {name} is already registered")
        configs[name] = cls

    return register


def get_config(name):
    return configs[name]()


class TestType(Enum):
    GLOBAL = "global"
    GLOBAL_COND = "global_cond"
    LOCAL_COND = "local_cond"


class ConceptType(Enum):
    DATASET = "dataset"
    CLASS = "concept"
    IMAGE = "image"


class Constants:
    WORKDIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    )
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"


class DataConfig(ConfigDict):
    def __init__(self, config_dict={}):
        super().__init__()
        self.dataset = config_dict.get("dataset", None)
        self.clip_backbone = config_dict.get("clip_backbone", None)
        self.num_concepts = config_dict.get("num_concepts", None)


class SpliceConfig(ConfigDict):
    def __init__(self, config_dict={}):
        super().__init__()
        self.vocab = config_dict.get("vocab", None)
        self.vocab_size = config_dict.get("vocab_size", None)
        self.l1_penalty = config_dict.get("l1_penalty", None)


class PCBMConfig(ConfigDict):
    def __init__(self, config_dict=None):
        super().__init__()
        self.alpha = config_dict.get("alpha", None)
        self.l1_ratio = config_dict.get("l1_ratio", None)


class cKDEConfig(ConfigDict):
    def __init__(self, config_dict={}):
        super().__init__()
        self.metric = config_dict.get("metric", None)
        self.scale_method = config_dict.get("scale_method", None)
        self.scale = config_dict.get("scale", None)


class TestingConfig(ConfigDict):
    def __init__(self, config_dict={}):
        super().__init__()
        self.significance_level = config_dict.get("significance_level", None)
        self.fdr_control = config_dict.get("fdr_control", None)
        self.wealth = config_dict.get("wealth", None)
        self.bet = config_dict.get("bet", None)
        self.kernel = config_dict.get("kernel", None)
        self.kernel_scale_method = config_dict.get("kernel_scale_method", None)
        self.kernel_scale = config_dict.get("kernel_scale", None)
        self.tau_max = config_dict.get("tau_max", None)
        self.r = config_dict.get("r", None)


class Config(ConfigDict):
    def __init__(self, config_dict={}):
        super().__init__()
        self.name = config_dict.get("name", None)
        self.data = DataConfig(config_dict.get("data", {}))
        self.splice = SpliceConfig(config_dict.get("splice", {}))
        self.pcbm = PCBMConfig(config_dict.get("pcbm", {}))
        self.ckde = cKDEConfig(config_dict.get("ckde", {}))
        self.testing = TestingConfig(config_dict.get("testing", {}))

    def freeze(self):
        return FrozenConfigDict(self)
