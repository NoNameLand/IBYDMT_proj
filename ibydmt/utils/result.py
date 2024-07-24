import os
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from ibydmt.utils.config import Config
from ibydmt.utils.config import Constants as c


class TestingResults:
    def __init__(self, config: Config, test_type: str, concept_type: str):
        self.name = config.name.lower()
        self.kernel = config.testing.kernel
        self.kernel_scale = config.testing.kernel_scale
        self.tau_max = config.testing.tau_max
        self.fdr_control = config.testing.fdr_control
        self.ckde_scale_method = config.ckde.scale_method
        self.ckde_scale = config.ckde.scale

        self.test_type = test_type
        self.concept_type = concept_type

        self.idx = []
        self.cardinality = []
        self.class_name = []
        self.concept = []
        self.rejected = []
        self.tau = []
        self.df = None

    def state_path(self, workdir: str = c.WORKDIR):
        state_dir = os.path.join(workdir, "results", self.name, self.test_type)
        os.makedirs(state_dir, exist_ok=True)

        state_name = (
            f"{self.kernel}_{self.kernel_scale}_{self.tau_max}_{self.concept_type}"
        )
        if self.fdr_control:
            state_name = f"{state_name}_fdr"
        if "cond" in self.test_type:
            state_name = f"{state_name}_{self.ckde_scale_method}_{self.ckde_scale}"
        return os.path.join(state_dir, f"{state_name}.parquet")

    def add(
        self,
        rejected: Iterable[bool],
        tau: Iterable[int],
        class_name: str,
        concepts: Iterable[str],
        idx: Optional[int] = None,
        cardinality: Optional[int] = None,
    ):
        for _concept, _rejected, _tau in zip(concepts, rejected, tau):
            self.idx.append(idx)
            self.cardinality.append(cardinality)
            self.class_name.append(class_name)
            self.concept.append(_concept)
            self.rejected.append(_rejected)
            self.tau.append(_tau)

    @staticmethod
    def load(
        config: Config,
        test_type: str,
        concept_type: str,
        workdir: str = c.WORKDIR,
        config_kw: Optional[Dict[str, Any]] = None,
    ):
        def _set(dict, key, value):
            keys = key.split(".")
            if len(keys) == 1:
                dict[keys[0]] = value
            else:
                _set(dict[keys[0]], ".".join(keys[1:]), value)

        config_dict = config.to_dict()
        for key, value in config_kw.items():
            _set(config_dict, key, value)
        config = Config(config_dict).freeze()

        results = TestingResults(config, test_type, concept_type)
        df = pd.read_parquet(results.state_path(workdir))
        results.df = df
        return results

    def save(self, workdir: str = c.WORKDIR):
        df = pd.DataFrame(
            {
                "idx": self.idx,
                "cardinality": self.cardinality,
                "class_name": self.class_name,
                "concept": self.concept,
                "rejected": self.rejected,
                "tau": self.tau,
            }
        )
        df.to_parquet(self.state_path(workdir))
