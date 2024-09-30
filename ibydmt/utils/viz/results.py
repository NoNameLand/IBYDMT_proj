import functools
import os
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ibydmt.tester import get_local_test_idx, get_test_classes
from ibydmt.utils.concepts import get_concepts
from ibydmt.utils.config import ConceptType, Config
from ibydmt.utils.config import Constants as c
from ibydmt.utils.config import TestType
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


class ConceptPostProcessor(Protocol):
    def __call__(
        self, sorted_concepts: Iterable[str], class_name: str, image_idx: int
    ) -> Iterable[str]:
        ...


def _viz_results(
    results: TestingResults,
    class_name: str,
    fdr_control: bool,
    ax: plt.Axes,
    image_idx: Optional[int] = None,
    cardinality: Optional[int] = None,
    concept_postprocessor: Optional[ConceptPostProcessor] = None,
):
    sorted_results = results.sort(
        class_name,
        image_idx=image_idx,
        cardinality=cardinality,
        fdr_control=fdr_control,
    )
    sorted_concepts = sorted_results["sorted_concepts"]
    sorted_rejected = sorted_results["sorted_rejected"]
    sorted_tau = sorted_results["sorted_tau"]

    if concept_postprocessor is not None:
        sorted_concepts = concept_postprocessor(sorted_concepts, class_name, image_idx)

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


def viz_global_results(
    config: Config,
    results: TestingResults,
    fdr_control: bool = True,
    show: bool = True,
    workdir=c.WORKDIR,
    concept_postprocessor: Optional[Callable] = None,
):
    figure_dir = os.path.join(workdir, "figures", config.name, results.test_type)
    os.makedirs(figure_dir, exist_ok=True)

    test_classes = get_test_classes(config, workdir=workdir)

    m = 5
    n = np.ceil(len(test_classes) / m).astype(int)
    _, axes = plt.subplots(n, m, figsize=(1.5 * 9, 16 / 2), gridspec_kw={"wspace": 1.0})
    for i, class_name in enumerate(test_classes):
        ax = axes[i // m, i % m]
        _viz_results(results, class_name, fdr_control, ax, concept_postprocessor)
        ax.set_title(class_name)
        if i == len(test_classes) - 1:
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        else:
            ax.legend().remove()

    plt.savefig(
        os.path.join(figure_dir, f"{results.backbone_name}_{results.concept_type}.pdf"),
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(figure_dir, f"{results.backbone_name}_{results.concept_type}.png"),
        bbox_inches="tight",
    )
    if show:
        plt.show()
    plt.close()


def viz_local_results(
    config: Config,
    results: TestingResults,
    cardinality: int = 1,
    fdr_control: bool = True,
    show: bool = True,
    workdir=c.WORKDIR,
    concept_postprocessor: Optional[Callable] = None,
):
    figure_dir = os.path.join(workdir, "figures", config.name, results.test_type)
    image_figure_dir = os.path.join(figure_dir, "images")
    backbone_figure_dir = os.path.join(figure_dir, results.backbone_name)
    os.makedirs(image_figure_dir, exist_ok=True)
    os.makedirs(backbone_figure_dir, exist_ok=True)

    dataset = get_dataset(config, train=False, workdir=workdir)
    test_idx = get_local_test_idx(config, workdir=workdir)

    for class_name, class_text_idx in test_idx.items():
        class_name_safe = class_name.lower().replace(" ", "_")

        for image_idx in class_text_idx:
            image, _ = dataset[image_idx]

            _, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(image)
            ax.axis("off")
            plt.savefig(
                os.path.join(
                    image_figure_dir,
                    f"{class_name_safe}_{image_idx}.pdf",
                ),
                bbox_inches="tight",
            )
            plt.savefig(
                os.path.join(
                    image_figure_dir,
                    f"{class_name_safe}_{image_idx}.png",
                ),
                bbox_inches="tight",
            )
            if show:
                plt.show()
            plt.close()

            _, ax = plt.subplots(figsize=(9 / 4, 16 / 4))
            _viz_results(
                results,
                class_name,
                fdr_control,
                ax,
                image_idx=image_idx,
                cardinality=cardinality,
                concept_postprocessor=concept_postprocessor,
            )
            ax.set_title(results.backbone_name)
            plt.savefig(
                os.path.join(
                    backbone_figure_dir,
                    f"{class_name_safe}_{image_idx}_{results.backbone_name}.pdf",
                ),
                bbox_inches="tight",
            )
            plt.savefig(
                os.path.join(
                    backbone_figure_dir,
                    f"{class_name_safe}_{image_idx}_{results.backbone_name}.png",
                ),
                bbox_inches="tight",
            )
            if show:
                plt.show()
            plt.close()


def viz_results(
    config: Config,
    test_type: str,
    concept_type: str,
    fdr_control: bool = True,
    show: bool = True,
    workdir=c.WORKDIR,
    cardinality: Optional[int] = None,
    results_kw: Optional[Mapping[str, Any]] = None,
):
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
    if test_type == TestType.LOCAL_COND.value:
        viz_fn = functools.partial(viz_local_results, cardinality=cardinality)

    concept_postprocessor = None
    if config.data.dataset.lower() == "awa2":

        def awa2_concept_postprocessor(
            sorted_concepts: Iterable[str], class_name: str, image_idx: int
        ) -> Iterable[str]:
            assert concept_type == ConceptType.CLASS.value, ValueError(
                "AwA2 concept postprocessor only support class-level concepts."
            )
            _, concepts = get_concepts(
                config, concept_class_name=class_name, workdir=workdir
            )

            return [
                f"{c} (p)" if c in concepts[:10] else f"{c} (a)"
                for c in sorted_concepts
            ]

        concept_postprocessor = awa2_concept_postprocessor
    if config.data.dataset.lower() == "cub":

        def cub_concept_postprocessor(
            sorted_concepts: Iterable[str], class_name: str, image_idx: int
        ) -> Iterable[str]:
            assert concept_type == ConceptType.IMAGE.value, ValueError(
                "CUB concept postprocessor only support image-level concepts."
            )

            def _attribute_to_human_readable(attribute: str) -> str:
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

            _, concepts = get_concepts(
                config, concept_image_idx=image_idx, workdir=workdir
            )

            return [
                f"{_attribute_to_human_readable(c)} ({'p' if c in concepts[:7] else 'a'})"
                for c in sorted_concepts
            ]

        concept_postprocessor = cub_concept_postprocessor

    if config.data.dataset.lower() == "imagenette":

        def imagenette_concept_postprocessor(
            sorted_concepts: Iterable[str], class_name: str, image_idx: int
        ):
            assert concept_type == ConceptType.IMAGE.value, ValueError(
                "Imagenette concept postprocessor only support image-level concepts."
            )

            _, concepts = get_concepts(
                config, concept_image_idx=image_idx, workdir=workdir
            )

            return [f"{c} (*)" if c in concepts[-4:] else c for c in sorted_concepts]

        concept_postprocessor = imagenette_concept_postprocessor

    if config.data.dataset.lower() == "cub_vip":

        def _attribute_to_human_readable(attribute: str) -> str:
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

        def cub_vip_concept_postprocessor(
            sorted_concepts: Iterable[str], class_name: str, image_idx: int
        ):
            assert concept_type == ConceptType.IMAGE.value, ValueError(
                "CUB VIP concept postprocessor only support image-level concepts."
            )

            _, concepts = get_concepts(
                config, concept_image_idx=image_idx, workdir=workdir
            )

            return [
                r"%s $(q_{%d})$"
                % (_attribute_to_human_readable(c), concepts.index(c) + 1)
                for c in sorted_concepts
            ]

        concept_postprocessor = cub_vip_concept_postprocessor

    for result in results:
        viz_fn(
            config,
            result,
            fdr_control=fdr_control,
            show=show,
            workdir=workdir,
            concept_postprocessor=concept_postprocessor,
        )


def viz_global_tau(
    config: Config,
    test_type: str,
    concept_type: str,
    workdir=c.WORKDIR,
    results_kw: Optional[Mapping[str, Any]] = None,
):
    assert test_type in [TestType.GLOBAL.value, TestType.GLOBAL_COND.value]
    figure_dir = os.path.join(workdir, "figures", config.name, test_type)

    test_classes = get_test_classes(config, workdir=workdir)

    tau_configs = config.sweep(["testing.tau_max"])
    results = {
        tau_config.testing.tau_max: TestingResults.load(
            tau_config, test_type, concept_type, workdir=workdir, results_kw=results_kw
        )
        for tau_config in tau_configs
    }
    assert len(set([_results.backbone_name for _results in results.values()])) == 1

    _, axes = plt.subplots(2, 5, figsize=(1.5 * 9, 16 / 2), gridspec_kw={"wspace": 1.0})
    for class_idx, class_name in enumerate(test_classes):
        concept_class_name = None
        if concept_type == ConceptType.CLASS.value:
            concept_class_name = class_name

        _, concepts = get_concepts(
            config, concept_class_name=concept_class_name, workdir=workdir
        )

        class_ranks = np.empty((len(tau_configs), len(concepts)))
        for t, _results in enumerate(results.values()):
            sorted_results = _results.sort(class_name)
            sorted_concepts = sorted_results["sorted_concepts"]
            class_ranks[t, [concepts.index(c) for c in sorted_concepts]] = (
                np.arange(len(concepts)) + 1
            )

        ax = axes[class_idx // 5, class_idx % 5]
        palette = sns.color_palette(
            "ch:s=.25,rot=-.25", n_colors=len(concepts)
        ).as_hex()[::-1]
        for rank in range(len(concepts)):
            concept_idx = class_ranks[-1].tolist().index(rank + 1)
            concept_ranks = class_ranks[:, concept_idx]

            y = len(concepts) - concept_ranks + 1
            ax.plot(range(len(tau_configs)), y, color=palette[rank])
            ax.annotate(
                concepts[concept_idx],
                (1.03, y[-1]),
                xycoords=("axes fraction", "data"),
                va="center",
            )

        ax.set_xlabel(r"$\tau^{\max}$")
        ax.set_ylabel("Rank")
        ax.set_xlim(0, len(tau_configs) - 1)
        ax.set_xticks(range(len(tau_configs)))
        ax.set_xticklabels(list(results.keys()))
        ax.set_yticks(range(1, len(concepts) + 1))
        ax.set_yticklabels(ax.get_yticks()[::-1])
        ax.set_title(class_name)

    plt.savefig(
        os.path.join(figure_dir, f"{_results.backbone_name}_{concept_type}_tau.pdf"),
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(figure_dir, f"{_results.backbone_name}_{concept_type}_tau.png"),
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


def viz_local_backbone(
    config: Config,
    concept_type: str,
    cardinality: int = 1,
    show: bool = True,
    workdir=c.WORKDIR,
    results_kw: Optional[Mapping[str, Any]] = None,
):
    test_type = TestType.LOCAL_COND.value
    figure_dir = os.path.join(workdir, "figures", config.name, test_type)

    test_idx = get_local_test_idx(config, workdir=workdir)

    backbone_configs = config.sweep(["data.backbone"])
    results = {
        backbone_config.data.backbone: TestingResults.load(
            backbone_config,
            test_type,
            concept_type,
            workdir=workdir,
            results_kw=results_kw,
        )
        for backbone_config in backbone_configs
    }

    for class_name, class_test_idx in test_idx.items():
        class_name_safe = class_name.lower().replace(" ", "_")

        for image_idx in class_test_idx:
            _, concepts = get_concepts(
                config, concept_image_idx=image_idx, workdir=workdir
            )

            class_ranks = np.empty((len(backbone_configs), len(concepts)))
            for b, _results in enumerate(results.values()):
                sorted_results = _results.sort(
                    class_name, image_idx=image_idx, cardinality=cardinality
                )
                sorted_concepts = sorted_results["sorted_concepts"]
                class_ranks[b, [concepts.index(c) for c in sorted_concepts]] = (
                    np.arange(len(concepts)) + 1
                )

            _, ax = plt.subplots(figsize=(9 / 2, 16 / 4))
            palette = sns.color_palette(
                "ch:s=.25,rot=-.25", n_colors=len(concepts)
            ).as_hex()[::-1]
            for rank in range(len(concepts)):
                concept_idx = class_ranks[-1].tolist().index(rank + 1)
                concept_ranks = class_ranks[:, concept_idx]

                y = len(concepts) - concept_ranks + 1
                ax.plot(range(len(backbone_configs)), y, color=palette[rank])
                ax.annotate(
                    concepts[concept_idx],
                    (1.03, y[-1]),
                    xycoords=("axes fraction", "data"),
                    va="center",
                )

            ax.set_xlabel("")
            ax.set_ylabel("Rank")
            ax.set_xlim(0, len(backbone_configs) - 1)
            ax.set_xticks(range(len(backbone_configs)))
            ax.set_xticklabels(list(results.keys()), rotation=45, ha="right")
            ax.set_yticks(range(1, len(concepts) + 1))
            ax.set_yticklabels(ax.get_yticks()[::-1])

            plt.savefig(
                os.path.join(figure_dir, f"{class_name_safe}_{image_idx}_backbone.pdf"),
                bbox_inches="tight",
            )
            plt.savefig(
                os.path.join(figure_dir, f"{class_name_safe}_{image_idx}_backbone.png"),
                bbox_inches="tight",
            )
            if show:
                plt.show()
            plt.close()
