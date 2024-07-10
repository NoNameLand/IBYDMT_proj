import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from counting_lib import CLASS_NAME, CONCEPTS, DIGIT_NAMES, CountingDataset
from notebooks.viz_results_utils import _viz_sorted_concepts, kernel_colors


def viz_dist(dataset, predictions, figure_dir):
    digits = pd.DataFrame(dataset.digits, columns=DIGIT_NAMES)

    gt_target = digits[CLASS_NAME].values
    pred_target = predictions[CLASS_NAME].values

    _, axes = plt.subplots(1, len(CONCEPTS), figsize=(16, 9 / 4))
    for i, concept in enumerate(CONCEPTS):
        gt_concept = digits[concept].values
        pred_concept = predictions[concept].values

        ax = axes[i]
        sns.scatterplot(
            x=gt_concept, y=gt_target, ax=ax, alpha=0.05, label="ground truth"
        )
        xlims, ylims = ax.get_xlim(), ax.get_ylim()
        sns.kdeplot(
            x=pred_concept,
            y=pred_target,
            ax=ax,
            levels=4,
            cut=0,
            bw_method=0.4,
            alpha=0.5,
            color="red",
            label="predictions",
        )

        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)
        ax.set_xlabel(concept)
        if concept in ["blue zeros", "orange threes"]:
            ax.set_xticks([0, 1, 2])
        if concept == "green fives":
            ax.set_xticks([1, 2, 3])
        elif concept in ["blue twos", "purple sevens"]:
            ax.set_xticks([1, 2])

        if i == 0:
            ax.set_ylabel(CLASS_NAME)
            ax.set_yticks([2, 3])
        else:
            ax.set_ylabel("")
            ax.set_yticks([])

    plt.savefig(os.path.join(figure_dir, "dist.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(figure_dir, "dist.png"), bbox_inches="tight")
    plt.show()
    plt.close()


def viz_results(results_dir, kernel, figure_dir):
    tau_max = [100, 200, 400, 800, 1600]
    results = {
        _tau_max: np.load(
            os.path.join(results_dir, f"{kernel}_{_tau_max}.npy"), allow_pickle=True
        )
        for _tau_max in tau_max
    }

    def get(results):
        rejected, tau = np.empty((len(CONCEPTS))), np.empty((len(CONCEPTS)))
        for data in results:
            concept = data["concept"]
            concept_idx = CONCEPTS.index(concept)

            _rejected = data["rejected"]
            _tau = data["tau"]

            rejected[concept_idx] = np.mean(_rejected)
            tau[concept_idx] = np.mean(_tau)

        return rejected, tau

    ranks = np.empty((len(tau_max), len(CONCEPTS)))
    for t, _results in enumerate(results.values()):
        _, tau = get(_results)
        tau /= tau_max[t]

        sorted_idx = np.argsort(tau)
        for j, idx in enumerate(sorted_idx):
            ranks[t, idx] = j + 1

    _, axes = plt.subplots(1, 2, figsize=(9 / 2, 16 / 8), gridspec_kw={"wspace": 0.7})
    ax = axes[0]
    rejected, tau = get(results[800])
    tau /= 800
    _viz_sorted_concepts(
        CONCEPTS, rejected, tau, ax, significance_level=0.05, **kernel_colors[kernel]
    )
    ax.set_title("red threes")

    ax = axes[1]
    palette = sns.color_palette("ch:s=.25,rot=-.25", n_colors=len(CONCEPTS)).as_hex()
    palette = palette[::-1]
    for rank in range(len(CONCEPTS)):
        concept_idx = ranks[-1].tolist().index(rank + 1)
        concept_ranks = ranks[:, concept_idx]

        y = len(CONCEPTS) - concept_ranks + 1
        ax.plot(range(len(tau_max)), y, color=palette[rank])
        ax.annotate(
            CONCEPTS[concept_idx],
            (1.03, y[-1]),
            xycoords=("axes fraction", "data"),
            va="center",
        )

    ax.set_xlabel(r"$\tau^{\max}$")
    ax.set_ylabel("Rank")
    ax.set_xlim(0, len(tau_max) - 1)
    ax.set_xticks(range(len(tau_max)))
    ax.set_xticklabels(tau_max)
    ax.set_yticks(range(1, len(CONCEPTS) + 1))
    ax.set_yticklabels(ax.get_yticks()[::-1])
    ax.set_title("red threes")

    plt.savefig(os.path.join(figure_dir, f"results_{kernel}.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(figure_dir, f"results_{kernel}.png"), bbox_inches="tight")
    plt.show()
    plt.close()


def viz_local_results(results_dir, gradcam_dir, figure_dir):
    test_idx, cond_idx = 2, 1
    test_concept, cond_concept = CONCEPTS[test_idx], CONCEPTS[cond_idx]

    dataset = CountingDataset(train=False)

    kernels, tau_max = ["linear", "rbf"], 400
    results = {
        kernel: np.load(
            os.path.join(results_dir, f"{kernel}_{tau_max}.npy"), allow_pickle=True
        )
        for kernel in kernels
    }

    viz_flag = [False, False, False]
    summary = {0: [], 1: [], 2: []}

    idx = np.load(os.path.join(results_dir, "idx.npy"))
    for _idx in idx:
        image, digits = dataset[_idx]

        test_digit = np.round(digits[test_idx]).astype(int)
        cond_digit = np.round(digits[cond_idx]).astype(int)

        if viz_flag[cond_digit]:
            continue

        explanation = np.load(os.path.join(gradcam_dir, f"{_idx}.npy"))

        _, axes = plt.subplots(1, 3, figsize=(16 / 2, 9 / 4))
        ax = axes[0]
        ax.imshow(image.squeeze().permute(1, 2, 0))
        ax.axis("off")
        ax.set_title(f"Conditioning concept\n{cond_concept} = {cond_digit}")

        ax = axes[1]
        ax.imshow(image.squeeze().permute(1, 2, 0))
        ax.axis("off")
        ax.imshow(explanation, cmap="bwr", alpha=0.5, vmin=-1, vmax=1)
        ax.set_title("Grad-CAM explanation")

        ax = axes[2]
        for kernel in kernels:
            linecolor, barcolor = kernel_colors[kernel].values()

            kernel_results = results[kernel]

            for data in kernel_results:
                if data["idx"] != _idx:
                    continue
                else:
                    break

            rejected, tau = np.mean(data["rejected"]), np.mean(data["tau"])
            tau /= tau_max

            ax.bar(
                x=kernel, height=tau, alpha=0.8, color=barcolor, label=f"Rejection time"
            )
            ax.scatter(x=kernel, y=rejected, color=linecolor, label=f"Rejection rate")
            ax.axhline(0.05, color="black", linestyle="--")
            ax.legend()

        ax.set_title(f"Local conditional importance of\n{test_concept} = {test_digit}")

        plt.savefig(
            os.path.join(figure_dir, f"{cond_concept}_{_idx}.pdf"), bbox_inches="tight"
        )
        plt.savefig(
            os.path.join(figure_dir, f"{cond_concept}_{_idx}.png"), bbox_inches="tight"
        )
        plt.show()
        plt.close()
