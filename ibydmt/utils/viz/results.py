import os
from typing import Any, Callable, Iterable, Mapping, Optional

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
    fdr_control: bool,
    ax: plt.Axes,
    concept_postprocessor: Optional[Callable],
):
    _, sorted_concepts, sorted_rejected, sorted_tau = results.sort(
        class_name, fdr_control=fdr_control
    )

    if concept_postprocessor is not None:
        sorted_concepts = concept_postprocessor(class_name, sorted_concepts)

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
        label="significance level",
        zorder=4,
        linewidth=1.5,
    )
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks(sorted_concepts)
    ax.set_title(class_name)


def viz_global_results(
    config: Config,
    results: TestingResults,
    fdr_control: bool = True,
    concept_postprocessor: Optional[Callable] = None,
):
    test_classes = get_test_classes(config)

    m = 5
    n = np.ceil(len(test_classes) / m).astype(int)
    _, axes = plt.subplots(n, m, figsize=(1.5 * 9, 16 / 2), gridspec_kw={"wspace": 1.0})
    for i, class_name in enumerate(test_classes):
        ax = axes[i // m, i % m]

        _viz_results(results, class_name, fdr_control, ax, concept_postprocessor)
        if i == len(test_classes) - 1:
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        else:
            ax.legend().remove()


def viz_results(
    config: Config,
    test_type: str,
    concept_type: str,
    fdr_control: bool = True,
    results_kw: Optional[Mapping[str, Any]] = None,
    workdir=c.WORKDIR,
):
    figure_dir = os.path.join(workdir, "figures", config.name, test_type)
    os.makedirs(figure_dir, exist_ok=True)

    sweep_backbone = True
    if results_kw.get("data.backbone", None) is not None:
        sweep_backbone = False

    if sweep_backbone:
        backbone_configs = config.sweep(["data.backbone"])
        results = [
            TestingResults.load(
                backbone_config,
                test_type,
                concept_type,
                workdir=workdir,
                results_kw=results_kw,
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
    if test_type == TestType.GLOBAL_COND.value:
        viz_fn = viz_global_results
    # if test_type == TestType.LOCAL_COND.value:
    #     viz_fn = viz_local_results

    concept_postprocessor = None
    if config.data.dataset.lower() == "awa2":

        def awa2_concept_postprocessor(
            class_name: str, sorted_concepts: Iterable[str]
        ) -> Iterable[str]:
            concept_class_name = None
            if concept_type == ConceptType.CLASS.value:
                concept_class_name = class_name

            _, concepts = get_concepts(
                config, concept_class_name=concept_class_name, workdir=workdir
            )

            return [
                f"{c} (p)" if c in concepts[:10] else f"{c} (a)"
                for c in sorted_concepts
            ]

        concept_postprocessor = awa2_concept_postprocessor

    for result in results:
        viz_fn(
            config,
            result,
            fdr_control=fdr_control,
            concept_postprocessor=concept_postprocessor,
        )

        plt.savefig(
            os.path.join(figure_dir, f"{result.backbone_name}_{concept_type}.pdf"),
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(figure_dir, f"{result.backbone_name}_{concept_type}.png"),
            bbox_inches="tight",
        )
        plt.show()
        plt.close()
