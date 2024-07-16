import os
import pickle
from collections import UserDict

from ibydmt.utils.config import Config


class DictionaryAccumulator(UserDict):
    def __init__(self):
        super().__init__()

    def append(self, key, value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)


class TestingResults:
    def __init__(self, config: Config, test_type, concept_type):
        self.name = config.name.lower()
        self.kernel = config.testing.kernel
        self.kernel_scale = config.testing.kernel_scale
        self.tau_max = config.testing.tau_max
        self.fdr_control = config.testing.fdr_control
        self.ckde_scale_method = config.ckde.scale_method
        self.ckde_scale = config.ckde.scale

        self.test_type = test_type
        self.concept_type = concept_type

        self.rejected = DictionaryAccumulator()
        self.tau = DictionaryAccumulator()

    def add(self, rejected, tau, class_name, concepts):
        for _concept, _rejected, _tau in zip(concepts, rejected, tau):
            key = (class_name, _concept)
            self.rejected.append(key, _rejected)
            self.tau.append(key, _tau)

    def state_path(self, workdir):
        state_dir = os.path.join(workdir, "results", self.name, self.test_type)
        os.makedirs(state_dir, exist_ok=True)

        state_name = (
            f"{self.kernel}_{self.kernel_scale}_{self.tau_max}_{self.concept_type}"
        )
        if "cond" in self.test_type:
            state_name = f"{state_name}_{self.ckde_scale_method}_{self.ckde_scale}"
        return os.path.join(state_dir, f"{state_name}.pt")

    def save(self, workdir):
        with open(self.state_path(workdir), "wb") as f:
            pickle.dump((self.rejected, self.tau), f)
