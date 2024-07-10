configs = {}


def register_config(name):
    def register(cls):
        if name in configs:
            raise ValueError(f"Config {name} is already registered")
        configs[name] = cls

    return register


def get_config(name):
    return configs[name]()


class DataConfig:
    def __init__(self, **kwargs):
        self.dataset = kwargs.pop("dataset", None)
        self.clip_backbone = kwargs.pop("clip_backbone", None)
        self.num_concepts = kwargs.pop("num_concepts", None)


class SpliceConfig:
    def __init__(self, **kwargs):
        self.vocab = kwargs.pop("vocab", None)
        self.vocab_size = kwargs.pop("vocab_size", None)
        self.l1_penalty = kwargs.pop("l1_penalty", None)


class PCBMConfig:
    def __init__(self, **kwargs):
        self.alpha = kwargs.pop("alpha", None)
        self.l1_ratio = kwargs.pop("l1_ratio", None)


class cKDEConfig:
    def __init__(self, **kwargs):
        self.metric = kwargs.pop("metric", None)
        self.scale_method = kwargs.pop("scale_method", None)
        self.scale = kwargs.pop("scale", None)


class TestingConfig:
    def __init__(self, **kwargs):
        self.significance_level = kwargs.pop("significance_level", None)
        self.wealth = kwargs.pop("wealth", None)
        self.bet = kwargs.pop("bet", None)
        self.kernel = kwargs.pop("kernel", None)
        self.kernel_scale_method = kwargs.pop("kernel_scale_method", None)
        self.kernel_scale = kwargs.pop("kernel_scale", None)
        self.tau_max = kwargs.pop("tau_max", None)
        self.r = kwargs.pop("r", None)


class Config:
    def __init__(self, **kwargs):
        self.name = kwargs.pop("name", None)
        self.data = DataConfig(**kwargs)
        self.splice = SpliceConfig(**kwargs)
        self.pcbm = PCBMConfig(**kwargs)
        self.ckde = cKDEConfig(**kwargs)
        self.testing = TestingConfig(**kwargs)

    def items(self):
        items = []
        for key, value in self.__dict__.items():
            if hasattr(value, "__dict__"):
                items.extend(value.__dict__.items())
            else:
                items.append((key, value))
        return items
