import os
from typing import Any, Iterable, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ibydmt.tester import get_local_test_idx
from ibydmt.utils.concepts import get_concepts
from ibydmt.utils.config import Config
from ibydmt.utils.config import Constants as c
from ibydmt.utils.data import get_dataset
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


def sort(
    concepts: Iterable[str],
    rejected: Iterable[float],
    tau: Iterable[float],
    with_importance: bool = False,
    significance_level: Optional[float] = None,
):
    if with_importance:
        assert significance_level is not None
        important = rejected > significance_level

        important_sorted_idx = np.argsort(tau[important])
        unimportant_sorted_idx = np.argsort(tau[~important])
        sorted_idx = np.concatenate([important_sorted_idx, unimportant_sorted_idx])
    else:
        sorted_idx = np.argsort(tau)

    sorted_concepts = [concepts[idx] for idx in sorted_idx]
    sorted_rejected = rejected[sorted_idx]
    sorted_tau = tau[sorted_idx]
    return sorted_idx, sorted_concepts, sorted_rejected, sorted_tau


def viz_results(
    results: TestingResults,
    class_name,
    concepts: Optional[Iterable[str]] = None,
    idx: Optional[int] = None,
    cardinality: Optional[int] = None,
    fdr_control=False,
    show_importance=False,
    ax: Optional[plt.Axes] = None,
    linecolor=Colors.DEFAULT_LINECOLOR,
    barcolor=Colors.DEFAULT_BARCOLOR,
):
    if concepts is None:
        concepts, rejected, tau = results.get(
            class_name, idx=idx, cardinality=cardinality
        )
    else:
        if idx is not None:
            _concepts, rejected, tau = results.get(
                class_name, idx=idx, cardinality=cardinality
            )
            assert set(concepts) == set(_concepts)
            concepts = [f"(*) {c}" if c in concepts[-4:] else c for c in _concepts]
        else:
            raise NotImplementedError

    if fdr_control:
        _, fdr_rejected, fdr_tau = results.get(
            class_name, idx=idx, cardinality=cardinality, fdr_control=True
        )
        important = fdr_rejected > results.significance_level

        sorted_idx, sorted_concepts, sorted_fdr_rejected, sorted_fdr_tau = sort(
            concepts,
            fdr_rejected,
            fdr_tau,
            with_importance=show_importance,
            significance_level=results.significance_level,
        )
        sorted_important = important[sorted_idx]

        ax.plot(
            sorted_fdr_rejected,
            sorted_concepts,
            color=Colors.FDR_LINECOLOR,
            marker="o",
            linestyle="--",
            zorder=2,
            linewidth=1.5,
        )
        sns.barplot(
            x=sorted_fdr_tau,
            y=sorted_concepts,
            color=Colors.FDR_BARCOLOR,
            # hue=sorted_important,
            # palette={False: f"{Colors.FDR_BARCOLOR}20", True: Colors.FDR_BARCOLOR},
            alpha=0.8,
            ax=ax,
            zorder=0,
        )
    else:
        important = rejected > results.significance_level

        sorted_idx = np.argsort(tau)
        sorted_concepts = [concepts[idx] for idx in sorted_idx]

    sorted_tau = tau[sorted_idx]
    sorted_rejected = rejected[sorted_idx]
    sorted_important = important[sorted_idx]

    ax.plot(
        sorted_rejected,
        sorted_concepts,
        color=linecolor,
        marker="o",
        linestyle="--",
        # label="rejection rate",
        zorder=3,
        linewidth=1.5,
    )
    sns.barplot(
        x=sorted_tau,
        y=sorted_concepts,
        color=Colors.DEFAULT_BARCOLOR,
        # hue=sorted_important,
        # palette={False: f"{barcolor}20", True: barcolor},
        # label="rejection time",
        ax=ax,
        zorder=1,
    )

    ax.axvline(
        results.significance_level,
        color="black",
        linestyle="--",
        # label="significance level",
        zorder=4,
        linewidth=1.5,
    )
    ax.set_xticks(np.linspace(0, 1, 3))
    ax.set_yticks(sorted_concepts)
    ax.set_title(class_name)


def viz_global(
    config: Config,
    concept_type: str = "dataset",
    workdir: str = c.WORKDIR,
    results_kw: Optional[Mapping[str, Any]] = None,
    include_fdr_control: bool = False,
):
    figure_dir = os.path.join(workdir, "figures", config.name.lower(), "global")
    os.makedirs(figure_dir, exist_ok=True)

    results = TestingResults.load(
        config, "global", concept_type, workdir=workdir, results_kw=results_kw
    )

    dataset = get_dataset(config)
    classes = dataset.classes

    _, axes = plt.subplots(2, 5, figsize=(1.5 * 9, 16 / 2), gridspec_kw={"wspace": 0.7})
    for class_idx, class_name in enumerate(classes):
        ax = axes[class_idx // 5, class_idx % 5]
        viz_results(results, class_name, fdr_control=include_fdr_control, ax=ax)

    figure_name = (
        f"{concept_type}_{results.kernel}_{results.kernel_scale}_{results.tau_max}"
    )
    if results.fdr_control == True:
        figure_name = f"{figure_name}_fdr"
    # plt.savefig(os.path.join(figure_dir, f"{figure_name}.pdf"), bbox_inches="tight")
    # plt.savefig(os.path.join(figure_dir, f"{figure_name}.png"), bbox_inches="tight")
    plt.show()
    plt.close()


def viz_local_cond(
    config: Config,
    concept_type: str = "image",
    workdir: str = c.WORKDIR,
    results_kw: Optional[Mapping[str, Any]] = None,
    include_fdr_control: bool = False,
):
    figure_dir = os.path.join(workdir, "figures", config.name.lower(), "local_cond")
    os.makedirs(figure_dir, exist_ok=True)

    results = TestingResults.load(
        config, "local_cond", concept_type, workdir=workdir, results_kw=results_kw
    )

    dataset = get_dataset(config, train=False, workdir=workdir)
    test_idx = get_local_test_idx(config, workdir=workdir)
    cardinalities = config.testing.cardinalities

    for class_name, class_test_idx in test_idx.items():
        for idx in class_test_idx:
            image, _ = dataset[idx]
            _, concepts = get_concepts(config, workdir, concept_image_idx=idx)

            _, axes = plt.subplots(
                1,
                len(cardinalities) + 2,
                figsize=(25, 16 / 4),
                gridspec_kw={"wspace": 0.8},
            )

            ax = axes[0]
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(class_name)

            for k, cardinality in enumerate(cardinalities):
                ax = axes[k + 1]
                viz_results(
                    results,
                    class_name,
                    concepts=concepts,
                    idx=idx,
                    cardinality=cardinality,
                    ax=ax,
                    fdr_control=include_fdr_control,
                )
                ax.set_title(r"s = %d" % cardinality)

            plt.show()
            plt.close()
