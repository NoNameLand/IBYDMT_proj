import os

from ibydmt.utils.config import Constants as c

datasets = {}


def register_dataset(name):
    def register(cls):
        if name in datasets:
            raise ValueError(f"Dataset {name} is already registered")
        datasets[name] = cls

    return register


def get_dataset(config, train=True, transform=None, workdir=c.WORKDIR):
    name = config.data.dataset.lower()
    root = os.path.join(workdir, "data")
    return datasets[name](root, train=train, transform=transform)
