from typing import Any, Iterable, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ibydmt.tester import get_test_classes
from ibydmt.utils.concepts import get_concepts
from ibydmt.utils.config import ConceptType, Config
from ibydmt.utils.config import Constants as c
from ibydmt.utils.config import TestType
from ibydmt.utils.result import TestingResults


class Colors:
    DEFAULT_LINECOLOR = "#1f78b4"
    DEFAULT_BARCOLOR = "#a6cee3"
    FDR_LINECOLOR = "#33a02c"
    FDR_BARCOLOR = "#b2df8a"

    LINEAR_KERNEL_LINECOLOR = DEFAULT_LINECOLOR
    LINEAR_KERNEL_BARCOLOR = DEFAULT_BARCOLOR
    RBF_KERNEL_LINECOLOR = "#e31a1c"
    RBF_KERNEL_BARCOLOR = "#fb9a99"


def _viz_results(
    results: TestingResults,
    class_name: str,
    concepts: Iterable[str],
    fdr_control: bool,
    ax: plt.Axes,
):
    _, rejected, tau = results.get(class_name, fdr_control=fdr_control)

    sorted_idx = np.argsort(tau)
    sorted_concepts = [concepts[idx] for idx in sorted_idx]
    sorted_rejected = rejected[sorted_idx]
    sorted_tau = tau[sorted_idx]

    ax.plot(
        sorted_rejected,
        sorted_concepts,
        color=Colors.DEFAULT_LINECOLOR,
        marker="o",
        linestyle="--",
        label="rejection rate",
        zorder=3,
        linewidth=1.5,
    )
    sns.barplot(
        x=sorted_tau,
        y=sorted_concepts,
        color=Colors.DEFAULT_BARCOLOR,
        label="rejection time",
        ax=ax,
        zorder=1,
    )

    ax.axvline(
        results.significance_level,
        color="black",
        linestyle="--",
        zorder=4,
        linewidth=1.5,
    )
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks(sorted_concepts)
    ax.set_title(class_name)


def viz_global_results(
    config: Config,
    concept_type: str,
    results: TestingResults,
    fdr_control: bool = True,
    workdir=c.WORKDIR,
):
    test_classes = get_test_classes(config)

    m = 5
    n = len(test_classes) // 2 + 1
    _, axes = plt.subplots(n, m, figsize=(1.5 * 9, 16 / 2), gridspec_kw={"wspace": 0.7})
    for i, class_name in enumerate(test_classes):
        ax = axes[i // m, i % m]

        concept_class_name = None
        if concept_type == ConceptType.CLASS.value:
            concept_class_name = class_name

        _, concepts = get_concepts(
            config, concept_class_name=concept_class_name, workdir=workdir
        )
        _viz_results(results, class_name, concepts, fdr_control, ax)


def viz_results(
    config: Config,
    test_type: str,
    concept_type: str,
    fdr_control: bool = True,
    results_kw: Optional[Mapping[str, Any]] = None,
    workdir=c.WORKDIR,
):
    sweep_backbone = True
    if results_kw.get("data.backbone", None) is not None:
        sweep_backbone = False

    if sweep_backbone:
        backbone_configs = config.sweep(["data.backbone"])
        results = [
            TestingResults.load(
                backbone_config, test_type, concept_type, workdir=workdir
            )
            for backbone_config in backbone_configs
        ]
    else:
        results = [
            TestingResults.load(
                config, test_type, concept_type, workdir=workdir, results_kw=results_kw
            )
        ]

    if test_type == TestType.GLOBAL.value:
        viz_fn = viz_global_results
    # if test_type == TestType.GLOBAL_COND.value:
    #     viz_fn = viz_global_cond_results
    # if test_type == TestType.LOCAL_COND.value:
    #     viz_fn = viz_local_results

    for result in results:
        viz_fn(config, concept_type, result, with_fdr=fdr_control, workdir=workdir)
