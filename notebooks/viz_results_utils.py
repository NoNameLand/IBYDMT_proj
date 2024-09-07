import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from concept_lib import get_concepts
from scipy.stats import spearmanr
from test_utils import TestingResults

from datasets import get_dataset
from models.pcbm import PCBM

kernel_colors = {
    "linear": {"linecolor": "#1f78b4", "barcolor": "#a6cee3"},
    "rbf": {"linecolor": "#e31a1c", "barcolor": "#fb9a99"},
}


def _viz_sorted_concepts(
    concepts,
    rejected,
    tau,
    ax,
    significance_level=None,
    linecolor="#1f78b4",
    barcolor="#a6cee3",
):
    sorted_idx = np.argsort(tau)
    sorted_tau = tau[sorted_idx]
    sorted_rejection = rejected[sorted_idx]
    sorted_concepts = [concepts[idx] for idx in sorted_idx]

    ax.plot(
        sorted_rejection,
        sorted_concepts,
        color=linecolor,
        marker="o",
        linestyle="--",
        label="rejection rate",
    )
    sns.barplot(
        x=sorted_tau,
        y=sorted_concepts,
        color=barcolor,
        alpha=0.8,
        label="rejection time",
        ax=ax,
    )
    if significance_level is not None:
        ax.axvline(
            significance_level,
            color="black",
            linestyle="--",
            label="significance level",
        )
    ax.set_xticks(np.linspace(0, 1, 3))
    ax.set_yticks(sorted_concepts)

    return sorted_idx, sorted_concepts


def _viz_sorted_pcbm(pcbm, class_idx, ax, barcolor="#a6cee3"):
    concepts = pcbm.concepts

    weights = pcbm.weights()
    class_weights = weights[class_idx]

    sorted_idx = np.argsort(np.abs(class_weights))[::-1]
    sorted_weights = class_weights[sorted_idx]
    sorted_concepts = [concepts[idx] for idx in sorted_idx]

    sorted_weights = sorted_weights / np.abs(sorted_weights).max()
    sns.barplot(
        x=sorted_weights,
        y=sorted_concepts,
        color=barcolor,
        alpha=0.8,
        label="weight",
        ax=ax,
    )
    ax.set_xticks(np.linspace(-1, 1, 5))
    ax.set_yticks(sorted_concepts)


def viz_results(results):
    test_type = results.test_type
    classes = results.classes
    concepts = results.concepts
    kernel = results.kernel

    for class_name in classes:
        rejected, tau = results.get(class_name)

        _, ax = plt.subplots(figsize=(9 / 4, 16 / 4))
        _viz_sorted_concepts(concepts, rejected, tau, ax, **kernel_colors[kernel])
        ax.set_title(f"{class_name}\n{test_type}")
        ax.legend()
        plt.show()
        plt.close()


def viz_global(config, workdir, concept_type=None, **results_kw):
    figure_dir = os.path.join(workdir, "figures", config.name.lower(), "global")
    os.makedirs(figure_dir, exist_ok=True)

    results = TestingResults.load(
        config, workdir, test_type="global", concept_type=concept_type, **results_kw
    )

    dataset = get_dataset(config)
    classes = dataset.classes

    _, axes = plt.subplots(2, 5, figsize=(1.5 * 9, 16 / 2), gridspec_kw={"wspace": 0.7})
    for class_idx, class_name in enumerate(classes):
        if concept_type == "dataset":
            _, concepts = get_concepts(config)
        elif concept_type == "class":
            _, concepts = get_concepts(config, concept_class_name=class_name)

        rejected, tau = results.get(class_name, concepts=concepts)

        ax = axes[class_idx // 5, class_idx % 5]
        _viz_sorted_concepts(
            concepts, rejected, tau, ax, significance_level=results.significance_level
        )
        ax.set_title(class_name)

    figure_name = (
        f"{concept_type}_{results.kernel}_{results.kernel_scale}_{results.tau_max}"
    )
    plt.savefig(os.path.join(figure_dir, f"{figure_name}.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(figure_dir, f"{figure_name}.png"), bbox_inches="tight")
    plt.show()
    plt.close()


def viz_global_cond(config, workdir, concept_type=None, **results_kw):
    figure_dir = os.path.join(workdir, "figures", config.name.lower(), "global_cond")
    os.makedirs(figure_dir, exist_ok=True)

    results = TestingResults.load(
        config,
        workdir,
        test_type="global_cond",
        concept_type=concept_type,
        **results_kw,
    )

    dataset = get_dataset(config)
    classes = dataset.classes

    tau_hist, weight_hist = [], []
    _, axes = plt.subplots(2, 5, figsize=(1.5 * 9, 16 / 2), gridspec_kw={"wspace": 0.7})
    for class_idx, class_name in enumerate(classes):
        if concept_type == "dataset":
            _, concepts = get_concepts(config)
            pcbm = PCBM.load_or_train(config, workdir)
        elif concept_type == "class":
            _, concepts = get_concepts(config, concept_class_name=class_name)
            pcbm = PCBM.load_or_train(config, workdir, concept_class_name=class_name)

        rejected, tau = results.get(class_name, concepts=concepts)

        assert pcbm.concepts == concepts
        pcbm_weights = pcbm.weights()[class_idx]
        pcbm_weights = np.abs(pcbm_weights)
        pcbm_weights = pcbm_weights / pcbm_weights.max()

        tau_hist.extend(tau)
        weight_hist.extend(pcbm_weights)

        ax = axes[class_idx // 5, class_idx % 5]
        sorted_idx, sorted_concepts = _viz_sorted_concepts(
            concepts, rejected, tau, ax, significance_level=results.significance_level
        )
        ax.plot(
            pcbm_weights[sorted_idx],
            sorted_concepts,
            color="lightcoral",
            marker="o",
            linestyle="--",
            alpha=0.5,
            label="PCBM weight",
        )
        ax.margins(y=5e-03)
        ax.set_yticks(sorted_concepts)
        ax.set_title(class_name)

    figure_name = f"{concept_type}_{results.kernel}_{results.kernel_scale}_{results.tau_max}_{results.ckde_scale_method}_{results.ckde_scale}"
    plt.savefig(os.path.join(figure_dir, f"{figure_name}.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(figure_dir, f"{figure_name}.png"), bbox_inches="tight")
    plt.show()
    plt.close()

    _, ax = plt.subplots(figsize=(3, 3))
    sns.regplot(
        x=tau_hist,
        y=weight_hist,
        ax=ax,
        scatter_kws={"alpha": 0.2, "s": 4},
        line_kws={"color": "red", "linestyle": "--", "linewidth": 1},
    )
    ax.set_xlabel("Rejection time")
    ax.set_ylabel("PCBM weight")
    plt.show()
    plt.close()


def viz_pcbm(config, workdir, concept_type=None):
    figure_dir = os.path.join(workdir, "figures", config.name.lower(), "global_cond")
    os.makedirs(figure_dir, exist_ok=True)

    dataset = get_dataset(config)
    classes = dataset.classes

    _, axes = plt.subplots(2, 5, figsize=(1.5 * 9, 16 / 2), gridspec_kw={"wspace": 0.7})
    for class_idx, class_name in enumerate(classes):
        if concept_type == "dataset":
            pcbm = PCBM.load_or_train(config, workdir)
        if concept_type == "class":
            pcbm = PCBM.load_or_train(config, workdir, concept_class_name=class_name)

        ax = axes[class_idx // 5, class_idx % 5]
        _viz_sorted_pcbm(pcbm, class_idx, ax)
        # ax.set_xlabel("PCBM weight")
        ax.set_title(class_name)

    figure_name = f"{concept_type}_pcbm"
    plt.savefig(os.path.join(figure_dir, f"{figure_name}.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(figure_dir, f"{figure_name}.png"), bbox_inches="tight")
    plt.show()
    plt.close()


def viz_global_tau(config, workdir, concept_type=None, **results_kw):
    figure_dir = os.path.join(workdir, "figures", config.name.lower(), "global")
    os.makedirs(figure_dir, exist_ok=True)

    dataset = get_dataset(config)
    classes = dataset.classes

    tau_max = config.testing.tau_max

    results = {
        _tau_max: TestingResults.load(
            config,
            workdir,
            test_type="global",
            concept_type=concept_type,
            tau_max=_tau_max,
            **results_kw,
        )
        for _tau_max in tau_max
    }

    _, axes = plt.subplots(
        2, 5, figsize=(1.5 * 9, 16 / 2), gridspec_kw={"hspace": 0.5, "wspace": 0.7}
    )
    for class_idx, class_name in enumerate(classes):
        if concept_type == "dataset":
            _, concepts = get_concepts(config)
        elif concept_type == "class":
            _, concepts = get_concepts(config, concept_class_name=class_name)

        class_ranks = np.empty((len(tau_max), len(concepts)))
        for t, _results in enumerate(results.values()):
            _, tau = _results.get(class_name, concepts=concepts)

            sorted_idx = np.argsort(tau)
            for j, idx in enumerate(sorted_idx):
                class_ranks[t, idx] = j + 1

        ax = axes[class_idx // 5, class_idx % 5]
        palette = sns.color_palette(
            "ch:s=.25,rot=-.25", n_colors=len(concepts)
        ).as_hex()[::-1]
        for rank in range(len(concepts)):
            concept_idx = class_ranks[-1].tolist().index(rank + 1)
            concept_ranks = class_ranks[:, concept_idx]

            y = len(concepts) - concept_ranks + 1
            ax.plot(range(len(tau_max)), y, color=palette[rank])
            ax.annotate(
                concepts[concept_idx],
                (1.03, y[-1]),
                xycoords=("axes fraction", "data"),
                va="center",
            )

        ax.set_xlabel(r"$\tau^{\max}$")
        ax.set_ylabel("Rank")
        ax.set_xlim(0, len(tau_max) - 1)
        ax.set_xticks(range(len(tau_max)))
        ax.set_xticklabels(tau_max)
        ax.set_yticks(range(1, len(concepts) + 1))
        ax.set_yticklabels(ax.get_yticks()[::-1])
        ax.set_title(class_name)

    figure_name = f"{concept_type}_{_results.kernel}_{_results.kernel_scale}_tau"
    plt.savefig(os.path.join(figure_dir, f"{figure_name}.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(figure_dir, f"{figure_name}.png"), bbox_inches="tight")
    plt.show()
    plt.close()


def viz_global_cond_tau(config, workdir, concept_type=None, **results_kw):
    figure_dir = os.path.join(workdir, "figures", config.name.lower(), "global_cond")
    os.makedirs(figure_dir, exist_ok=True)

    dataset = get_dataset(config)
    classes = dataset.classes

    tau_max = config.testing.tau_max

    results = {
        _tau_max: TestingResults.load(
            config,
            workdir,
            test_type="global_cond",
            concept_type=concept_type,
            tau_max=_tau_max,
            **results_kw,
        )
        for _tau_max in tau_max
    }

    _, axes = plt.subplots(
        2, 5, figsize=(1.5 * 9, 16 / 2), gridspec_kw={"hspace": 0.5, "wspace": 0.7}
    )
    for class_idx, class_name in enumerate(classes):
        if concept_type == "dataset":
            _, concepts = get_concepts(config)
        elif concept_type == "class":
            _, concepts = get_concepts(config, concept_class_name=class_name)

        class_ranks = np.empty((len(tau_max), len(concepts)))
        for t, _results in enumerate(results.values()):
            _, tau = _results.get(class_name, concepts=concepts)

            sorted_idx = np.argsort(tau)
            for j, idx in enumerate(sorted_idx):
                class_ranks[t, idx] = j + 1

        ax = axes[class_idx // 5, class_idx % 5]
        palette = sns.color_palette(
            "ch:s=.25,rot=-.25", n_colors=len(concepts)
        ).as_hex()[::-1]
        for rank in range(len(concepts)):
            concept_idx = class_ranks[-1].tolist().index(rank + 1)
            concept_ranks = class_ranks[:, concept_idx]

            y = len(concepts) - concept_ranks + 1
            ax.plot(range(len(tau_max)), y, color=palette[rank])
            ax.annotate(
                concepts[concept_idx],
                (1.03, y[-1]),
                xycoords=("axes fraction", "data"),
                va="center",
            )

        ax.set_xlabel(r"$\tau^{\max}$")
        ax.set_ylabel("Rank")
        ax.set_xlim(0, len(tau_max) - 1)
        ax.set_xticks(range(len(tau_max)))
        ax.set_xticklabels(tau_max)
        ax.set_yticks(range(1, len(concepts) + 1))
        ax.set_yticklabels(ax.get_yticks()[::-1])
        ax.set_title(class_name)

    figure_name = f"{concept_type}_{_results.kernel}_{_results.kernel_scale}_tau_{_results.ckde_scale_method}_{_results.ckde_scale}"
    plt.savefig(os.path.join(figure_dir, f"{figure_name}.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(figure_dir, f"{figure_name}.png"), bbox_inches="tight")
    plt.show()
    plt.close()


def viz_local(config, workdir, cardinalities=None, tau_max=None, **results_kw):
    figure_dir = os.path.join(workdir, "figures", config.name.lower(), "local_cond")
    os.makedirs(figure_dir, exist_ok=True)

    dataset = get_dataset(config, train=False)

    results = TestingResults.load(
        config,
        workdir,
        test_type="local_cond",
        concept_type="image",
        tau_max=tau_max,
        **results_kw,
    )
    significance_level = results.significance_level
    cardinalities = results.cardinalities

    test_idx = list(results._results.keys())
    for idx in test_idx:
        image, label = dataset[idx]
        class_name = dataset.classes[label]

        idx_figure_dir = os.path.join(figure_dir, f"{class_name}_{idx}")
        os.makedirs(idx_figure_dir, exist_ok=True)

        _, concepts = get_concepts(config, concept_image_idx=idx)
        _concepts = concepts.copy()
        _concepts[-4:] = [f"(*) {concept}" for concept in _concepts[-4:]]

        rejected, tau = results.get(
            class_name, concepts, idx=idx, cardinalities=cardinalities
        )

        _, axes = plt.subplots(
            1, len(cardinalities) + 2, figsize=(25, 16 / 4), gridspec_kw={"wspace": 0.8}
        )

        ax = axes[0]
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(class_name)

        for k, cardinality in enumerate(cardinalities):
            _rejected, _tau = rejected[:, k], tau[:, k]

            ax = axes[k + 1]
            _viz_sorted_concepts(
                _concepts, _rejected, _tau, ax, significance_level=significance_level
            )
            ax.get_legend().remove()
            ax.set_title(r"$s = %d$" % cardinality)

        idx_cardinality_ranks = np.empty((len(cardinalities), len(concepts)))
        for k, cardinality in enumerate(cardinalities):
            _tau = tau[:, k]

            sorted_idx = np.argsort(_tau)
            for j, idx in enumerate(sorted_idx):
                idx_cardinality_ranks[k, idx] = j + 1

        ax = axes[-1]
        palette = sns.color_palette("ch:s=.25,rot=-.25", n_colors=len(concepts))[::-1]
        for rank in range(len(concepts)):
            concept_idx = idx_cardinality_ranks[-1].tolist().index(rank + 1)
            concept_ranks = idx_cardinality_ranks[:, concept_idx]

            y = len(concepts) - concept_ranks + 1
            ax.plot(range(len(cardinalities)), y, color=palette[rank])
            ax.annotate(
                concepts[concept_idx],
                (1.03, y[-1]),
                xycoords=("axes fraction", "data"),
                va="center",
            )

        ax.set_xlabel(r"$s$")
        ax.set_ylabel("Rank")
        ax.set_xlim(0, len(cardinalities) - 1)
        ax.set_xticks(range(len(cardinalities)))
        ax.set_xticklabels(cardinalities)
        ax.set_yticks(range(1, len(concepts) + 1))
        ax.set_yticklabels(ax.get_yticks()[::-1])

        figure_name = os.path.join(
            f"{results.kernel}_{results.kernel_scale}_{results.tau_max}_{results.ckde_scale_method}_{results.ckde_scale}"
        )
        plt.savefig(
            os.path.join(idx_figure_dir, f"{figure_name}.pdf"), bbox_inches="tight"
        )
        plt.savefig(
            os.path.join(idx_figure_dir, f"{figure_name}.png"), bbox_inches="tight"
        )
        plt.show()
        plt.close()
