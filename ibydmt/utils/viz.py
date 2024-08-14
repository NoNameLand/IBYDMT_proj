import os
from typing import Any, Iterable, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ibydmt.classifiers import ZeroShotClassifier
from ibydmt.samplers import get_sampler
from ibydmt.tester import get_local_test_idx, get_test_classes, sample_random_subset
from ibydmt.utils.concept_data import get_dataset_with_concepts
from ibydmt.utils.concepts import get_concepts
from ibydmt.utils.config import ConceptType, Config
from ibydmt.utils.config import Constants as c
from ibydmt.utils.data import get_dataset
from ibydmt.utils.pcbm import PCBM
from ibydmt.utils.result import TestingResults


def cub_attribute_to_human_readable(attribute: str):
    first, second = attribute.split("::")
    part = " ".join(first.split("_")[1:-1])
    desc = second.replace("_", " ")

    if part in ["primary", "size"]:
        return f"is {desc}"
    if part == "bill" and "head" in desc:
        return f"bill {desc}"
    if part == "wing" and "wings" in desc:
        return desc.replace("-", " ")
    else:
        return f"{desc} {part}"


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
        _concepts, rejected, tau = results.get(
            class_name, idx=idx, cardinality=cardinality
        )
        assert set(concepts) == set(_concepts)

        if results.name == "imagenette":
            concepts = [f"(*) {c}" if c in concepts[-4:] else c for c in _concepts]
        elif results.name == "cub":
            human_concepts = [cub_attribute_to_human_readable(c) for c in _concepts]
            concepts = [
                f"(*) {human}" if c in concepts[-7:] else human
                for c, human in zip(_concepts, human_concepts)
            ]
        elif results.name == "awa2":
            concepts = [f"(*) {c}" if c in concepts[-10:] else c for c in _concepts]

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
        # sorted_important = important[sorted_idx]

        # ax.plot(
        #     sorted_fdr_rejected,
        #     sorted_concepts,
        #     color=Colors.FDR_LINECOLOR,
        #     marker="o",
        #     linestyle="--",
        #     zorder=2,
        #     linewidth=1.5,
        # )
        # sns.barplot(
        #     x=sorted_fdr_tau,
        #     y=sorted_concepts,
        #     color=Colors.FDR_BARCOLOR,
        #     # hue=sorted_important,
        #     # palette={False: f"{Colors.FDR_BARCOLOR}20", True: Colors.FDR_BARCOLOR},
        #     alpha=0.8,
        #     ax=ax,
        #     zorder=0,
        # )
    else:
        important = rejected > results.significance_level

        sorted_idx = np.argsort(tau)
        sorted_concepts = [concepts[idx] for idx in sorted_idx]

    sorted_tau = tau[sorted_idx]
    sorted_rejected = rejected[sorted_idx]
    sorted_important = important[sorted_idx]

    ax.plot(
        sorted_fdr_rejected,
        sorted_concepts,
        color=Colors.DEFAULT_LINECOLOR,
        marker="o",
        linestyle="--",
        label="rejection rate",
        zorder=3,
        linewidth=1.5,
    )
    sns.barplot(
        x=sorted_fdr_tau,
        y=sorted_concepts,
        color=Colors.DEFAULT_BARCOLOR,
        # hue=sorted_important,
        # palette={False: f"{barcolor}20", True: barcolor},
        label="rejection time",
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

    test_classes = get_test_classes(config, workdir=workdir)

    _, axes = plt.subplots(2, 5, figsize=(1.5 * 9, 16 / 2), gridspec_kw={"wspace": 0.7})
    for i, class_name in enumerate(test_classes):
        ax = axes[i // 5, i % 5]

        concept_class_name = None
        if concept_type == ConceptType.CLASS.value:
            concept_class_name = class_name

        _, concepts = get_concepts(
            config, concept_class_name=concept_class_name, workdir=workdir
        )

        viz_results(
            results,
            class_name,
            concepts=concepts,
            fdr_control=include_fdr_control,
            ax=ax,
        )

    figure_name = (
        f"{concept_type}_{results.kernel}_{results.kernel_scale}_{results.tau_max}"
    )
    if results.fdr_control == True:
        figure_name = f"{figure_name}_fdr"
    # plt.savefig(os.path.join(figure_dir, f"{figure_name}.pdf"), bbox_inches="tight")
    # plt.savefig(os.path.join(figure_dir, f"{figure_name}.png"), bbox_inches="tight")
    plt.show()
    plt.close()


def viz_global_cond(
    config: Config,
    concept_type: str = "dataset",
    workdir: str = c.WORKDIR,
    results_kw: Optional[Mapping[str, Any]] = None,
    include_fdr_control: bool = False,
):
    figure_dir = os.path.join(workdir, "figures", config.name.lower(), "global_cond")
    os.makedirs(figure_dir, exist_ok=True)

    results = TestingResults.load(
        config, "global_cond", concept_type, workdir=workdir, results_kw=results_kw
    )

    test_classes = get_test_classes(config, workdir=workdir)

    _, axes = plt.subplots(2, 5, figsize=(1.5 * 9, 16 / 2), gridspec_kw={"wspace": 0.7})
    for i, class_name in enumerate(test_classes):
        ax = axes[i // 5, i % 5]

        concept_class_name = None
        if concept_type == ConceptType.CLASS.value:
            concept_class_name = class_name

        _, concepts = get_concepts(
            config, concept_class_name=concept_class_name, workdir=workdir
        )

        viz_results(
            results,
            class_name,
            concepts=concepts,
            fdr_control=include_fdr_control,
            ax=ax,
        )


def viz_pcbm(config: Config, concept_type: str = "dataset", workdir: str = c.WORKDIR):
    dataset = get_dataset(config, workdir=workdir)
    test_classes = get_test_classes(config, workdir=workdir)

    _, axes = plt.subplots(2, 5, figsize=(1.5 * 9, 16 / 2), gridspec_kw={"wspace": 0.7})
    for i, class_name in enumerate(test_classes):
        ax = axes[i // 5, i % 5]

        class_idx = dataset.classes.index(class_name)

        concept_class_name = None
        if concept_type == ConceptType.CLASS.value:
            concept_class_name = class_name

        _, concepts = get_concepts(
            config, workdir=workdir, concept_class_name=class_name
        )
        pcbm = PCBM.load_or_train(
            config, workdir=workdir, concept_class_name=concept_class_name
        )
        weights = pcbm.weights()
        class_weights = weights[class_idx]

        sorted_idx = np.argsort(np.abs(class_weights))[::-1]
        sorted_concepts = [pcbm.concepts[idx] for idx in sorted_idx]
        sorted_concepts = [
            f"(*) {c}" if c in concepts[-10:] else c for c in sorted_concepts
        ]
        sorted_weights = class_weights[sorted_idx]
        sns.barplot(x=sorted_weights, y=sorted_concepts, ax=ax)
        ax.set_yticks(sorted_concepts)
        ax.set_title(class_name)

    plt.show()
    plt.close()


def viz_local_cond(
    config: Config,
    concept_type: str = "image",
    workdir: str = c.WORKDIR,
    results_kw: Optional[Mapping[str, Any]] = None,
    include_fdr_control: bool = False,
):
    results = TestingResults.load(
        config, "local_cond", concept_type, workdir=workdir, results_kw=results_kw
    )

    figure_dir = os.path.join(
        workdir, "figures", config.name.lower(), "local_cond", results.backbone_name
    )
    os.makedirs(figure_dir, exist_ok=True)

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
                ax.set_title(f"{results.backbone_name}\ns = {cardinality}")

            plt.savefig(
                os.path.join(figure_dir, f"{class_name}_{idx}.pdf"), bbox_inches="tight"
            )
            plt.savefig(
                os.path.join(figure_dir, f"{class_name}_{idx}.png"), bbox_inches="tight"
            )
            plt.show()
            plt.close()


def viz_cond_pdf(
    config: Config, concept_type: str = "image", workdir=c.WORKDIR, device=c.DEVICE
):
    m = int(1e03)

    dataset = get_dataset(config, train=False, workdir=workdir)

    test_idx = get_local_test_idx(config, workdir=workdir)
    for class_name, class_test_idx in test_idx.items():
        for idx in class_test_idx:
            image, _ = dataset[idx]

            _, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(class_name)
            plt.show()

            concept_class_name = None
            concept_image_idx = None
            if concept_type == ConceptType.CLASS.value:
                concept_class_name = class_name
            if concept_type == ConceptType.IMAGE.value:
                concept_image_idx = idx

            concept_dataset = get_dataset_with_concepts(
                config,
                train=False,
                workdir=workdir,
                concept_class_name=concept_class_name,
                concept_image_idx=concept_image_idx,
            )
            concepts = concept_dataset.concepts
            semantics = concept_dataset.semantics

            z = semantics[idx]
            test_concepts = concepts[:2] + concepts[-2:]

            sampler = get_sampler(
                config,
                concept_class_name=concept_class_name,
                concept_image_idx=concept_image_idx,
            )

            _, axes = plt.subplots(
                1,
                len(test_concepts),
                figsize=(16 / 2, 9 / 4),
                gridspec_kw={"wspace": 0.5},
            )
            for i, concept in enumerate(test_concepts):
                concept_idx = concepts.index(concept)
                cond_idx = list(set(range(len(z))) - set([concept_idx]))

                marginal = semantics[:, concept_idx]
                conditional = sampler.sample_concept(z, cond_idx, m=m)[:, concept_idx]

                ax = axes[i]
                sns.histplot(
                    x=marginal,
                    kde=True,
                    stat="density",
                    color="lightblue",
                    label="marginal",
                    ax=ax,
                )
                sns.histplot(
                    x=conditional,
                    kde=True,
                    stat="density",
                    color="lightcoral",
                    label="conditional",
                    ax=ax,
                )
                ax.set_xlabel(r"$Z_{\text{%s}}$" % concept)
                ax.legend()

        plt.show()
        plt.close()
        break


def viz_local_dist(
    config: Config, concept_type: str = "image", workdir=c.WORKDIR, device=c.DEVICE
):
    m = int(1e03)
    cardinalities = config.testing.cardinalities

    dataset = get_dataset(config, train=False, workdir=workdir)
    classes = dataset.classes

    classifier = ZeroShotClassifier.load_or_train(
        config, workdir=workdir, device=device
    )

    test_idx = get_local_test_idx(config, workdir=workdir)
    for class_name, class_test_idx in test_idx.items():
        class_idx = classes.index(class_name)

        for idx in class_test_idx:
            image, _ = dataset[idx]

            _, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(class_name)
            plt.show()

            concept_class_name = None
            concept_image_idx = None
            if concept_type == ConceptType.CLASS.value:
                concept_class_name = class_name
            if concept_type == ConceptType.IMAGE.value:
                concept_image_idx = idx

            concept_dataset = get_dataset_with_concepts(
                config,
                train=False,
                workdir=workdir,
                concept_class_name=concept_class_name,
                concept_image_idx=concept_image_idx,
            )
            concepts = concept_dataset.concepts
            semantics = concept_dataset.semantics

            z = semantics[idx]
            test_concepts = concepts[:2] + concepts[-2:]

            sampler = get_sampler(
                config,
                concept_class_name=concept_class_name,
                concept_image_idx=concept_image_idx,
            )

            _, axes = plt.subplots(
                len(test_concepts),
                len(cardinalities),
                figsize=(16 / 2, 9),
                gridspec_kw={"hspace": 0.8},
            )
            for i, concept in enumerate(test_concepts):
                concept_idx = concepts.index(concept)

                for j, cardinality in enumerate(cardinalities):
                    subset_idx = sample_random_subset(
                        concepts, concept_idx, cardinality
                    )

                    null_h = sampler.sample_embedding(z, subset_idx, m=m)
                    test_h = sampler.sample_embedding(
                        z, subset_idx + [concept_idx], m=m
                    )

                    null_y = classifier(null_h)[:, class_idx]
                    test_y = classifier(test_h)[:, class_idx]

                    ax = axes[i, j]
                    sns.histplot(
                        x=null_y,
                        kde=True,
                        stat="density",
                        color="lightblue",
                        label="null",
                        ax=ax,
                    )
                    sns.histplot(
                        x=test_y,
                        kde=True,
                        stat="density",
                        color="lightcoral",
                        label="test",
                        ax=ax,
                    )
                    ax.set_xlabel(r"$\widehat{Y}_{\text{%s}}$" % class_name)
                    # ax.set_xlim(0.05, 0.45)
                    # ax.set_xticks([0.10, 0.20, 0.30, 0.40])
                    ax.set_ylim(0, 16)
                    if j == 0:
                        ax.set_ylabel("Density")
                        ax.set_yticks([0, 5, 10, 15])
                    else:
                        ax.set_ylabel("")
                        ax.set_yticks([])
                    ax.set_title(
                        "\n".join([f"{concept}", f"cardinality = {cardinality}"])
                    )
                    ax.legend()

            plt.show()
            plt.close()
