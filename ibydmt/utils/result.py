import os
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from ibydmt.utils.config import Config
from ibydmt.utils.config import Constants as c


class TestingResults:
    def __init__(self, config: Config, test_type: str, concept_type: str):
        self.name = config.name.lower()
        self.backbone_name = config.backbone_name()
        self.significance_level = config.testing.significance_level
        self.kernel = config.testing.kernel
        self.kernel_scale = config.testing.kernel_scale
        self.tau_max = config.testing.tau_max
        self.ckde_scale_method = config.ckde.scale_method
        self.ckde_scale = config.ckde.scale

        self.test_type = test_type
        self.concept_type = concept_type

        self.class_name = []
        self.concept = []
        self.rejected = []
        self.tau = []
        self.fdr_control = []
        self.idx = []
        self.cardinality = []
        self.df: pd.DataFrame = None

    def state_path(self, workdir: str = c.WORKDIR):
        state_dir = os.path.join(workdir, "results", self.name, self.test_type)
        os.makedirs(state_dir, exist_ok=True)

        state_name = f"{self.backbone_name}_{self.kernel}_{self.kernel_scale}_{self.tau_max}_{self.concept_type}"
        if "cond" in self.test_type:
            state_name = f"{state_name}_{self.ckde_scale_method}_{self.ckde_scale}"
        return os.path.join(state_dir, f"{state_name}.parquet")

    def add(
        self,
        class_name: str,
        concepts: Iterable[str],
        rejected: Iterable[bool],
        tau: Iterable[int],
        fdr_control: bool = False,
        idx: Optional[int] = None,
        cardinality: Optional[int] = None,
    ):
        for _concept, _rejected, _tau in zip(concepts, rejected, tau):
            self.class_name.append(class_name)
            self.concept.append(_concept)
            self.rejected.append(_rejected)
            self.tau.append(_tau)
            self.fdr_control.append(fdr_control)
            self.idx.append(idx)
            self.cardinality.append(cardinality)

    @staticmethod
    def load(
        config: Config,
        test_type: str,
        concept_type: str,
        workdir: str = c.WORKDIR,
        results_kw: Optional[Mapping[str, Any]] = None,
    ):
        def _set(dict, key, value):
            keys = key.split(".")
            if len(keys) == 1:
                dict[keys[0]] = value
            else:
                _set(dict[keys[0]], ".".join(keys[1:]), value)

        config_dict = config.to_dict()
        if results_kw is not None:
            for key, value in results_kw.items():
                _set(config_dict, key, value)
        config = Config(config_dict)

        results = TestingResults(config, test_type, concept_type)
        df = pd.read_parquet(results.state_path(workdir))
        results.df = df
        return results

    def save(self, workdir: str = c.WORKDIR):
        df = pd.DataFrame(
            {
                "class_name": self.class_name,
                "concept": self.concept,
                "rejected": self.rejected,
                "tau": self.tau,
                "fdr_control": self.fdr_control,
                "idx": self.idx,
                "cardinality": self.cardinality,
            }
        )
        df.to_parquet(self.state_path(workdir))

    def get(
        self,
        class_name: str,
        idx: Optional[int] = None,
        cardinality: Optional[int] = None,
        fdr_control: bool = False,
        normalize_tau: bool = True,
    ):
        assert cardinality is not None if idx is not None else True, ValueError(
            "Cardinality must be provided with idx"
        )

        df_filter = self.df["class_name"] == class_name
        if idx is not None:
            df_filter &= self.df["idx"] == idx
        if cardinality is not None:
            df_filter &= self.df["cardinality"] == cardinality
        df_filter &= self.df["fdr_control"] == fdr_control

        results = self.df[df_filter]
        results = results.groupby(["concept"])[["rejected", "tau"]].mean()

        concepts = results.index.values.tolist()
        rejected = results["rejected"].values
        tau = results["tau"].values

        if normalize_tau:
            tau /= self.tau_max
        return concepts, rejected, tau

    def sort(
        self,
        class_name: str,
        idx: Optional[int] = None,
        cardinality: Optional[int] = None,
        normalize_tau: bool = True,
        fdr_control: bool = False,
        with_importance: bool = False,
    ):
        concepts, rejected, tau = self.get(
            class_name,
            idx=idx,
            cardinality=cardinality,
            fdr_control=fdr_control,
            normalize_tau=normalize_tau,
        )

        sorted_idx = np.argsort(tau)
        sorted_concepts = [concepts[idx] for idx in sorted_idx]
        sorted_rejected = rejected[sorted_idx]
        sorted_tau = tau[sorted_idx]

        output = (sorted_idx, sorted_concepts, sorted_rejected, sorted_tau)
        if with_importance:
            importance = (rejected > self.significance_level).astype(int)
            sorted_importance = importance[sorted_idx]

            output += (sorted_importance,)
        return output
