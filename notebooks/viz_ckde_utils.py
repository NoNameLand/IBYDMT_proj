import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import entropy, gaussian_kde

test_class_concepts = {
    "tench": ["fish", "fishing", "selling", "trumpet"],
    "English springer": ["spaniel", "exterior", "trumpet", "fore"],
    "cassette player": ["radio", "selling", "cathedral", "band"],
    "chainsaw": ["instrument", "brass", "trumpet", "flew"],
    "church": ["cathedral", "exterior", "trumpet", "fore"],
    "French horn": ["trumpet", "brass", "fish", "bass"],
    "garbage truck": ["band", "bass", "fishing", "jazz"],
    "gas pump": ["dispenser", "trumpet", "fishing", "exterior"],
    "golf ball": ["putting", "fore", "fish", "instrument"],
    "parachute": ["flew", "airshow", "cathedral", "instrument"],
}


def _sample_subset(concepts, concept_idx, cardinality):
    sample_idx = list(set(range(len(concepts))) - set([concept_idx]))
    return np.random.permutation(sample_idx)[:cardinality].tolist()


def viz_cond_pdf(model, z, class_name, concepts, figure_dir):
    test_concepts = test_class_concepts[class_name]

    _, axes = plt.subplots(1, len(test_concepts), figsize=(16 / 2, 9 / 5))
    for i, concept in enumerate(test_concepts):
        concept_idx = concepts.index(concept)
        cond_idx = list(set(range(len(concepts))) - set([concept_idx]))

        ckde, _ = model.kde(z, cond_idx)
        Zj = ckde.dataset.squeeze()
        null_Zj = ckde.resample(int(1e04)).squeeze()

        kde = gaussian_kde(Zj)

        pdf = kde.pdf(Zj)
        null_pdf = ckde.pdf(Zj)
        kl = entropy(pdf, qk=null_pdf)

        ax = axes[i]
        sns.histplot(
            Zj,
            ax=ax,
            kde=True,
            color="lightblue",
            alpha=0.5,
            stat="density",
            label="marginal",
        )
        sns.histplot(
            null_Zj,
            ax=ax,
            kde=True,
            color="lightcoral",
            alpha=0.5,
            stat="density",
            label="conditional",
        )
        ax.set_xlabel(r"$Z_{\text{%s}}$" % concept)
        ax.set_xlim(0.05, 0.35)
        ax.set_xticks([0.10, 0.20, 0.30])
        ax.set_ylim(0, 40)
        if i == 0:
            ax.set_ylabel("Density")
            ax.set_yticks([0, 10, 20, 30, 40])
        else:
            ax.set_ylabel("")
            ax.set_yticks([])
        ax.set_title(
            "\n".join(
                [
                    f"KL = {kl:.2f}",
                    r"$n_{\text{eff}} = %d$" % model.scale,
                ]
            )
        )
        ax.legend()

    plt.savefig(
        os.path.join(figure_dir, f"cond_z_{model.scale}.pdf"),
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(figure_dir, f"cond_z_{model.scale}.png"),
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


def viz_local_dist(model, classifier, z, class_name, concepts, figure_dir):
    m = int(1e03)
    cardinalities = [1, 2, 4, 8]

    test_concepts = test_class_concepts[class_name]

    _, axes = plt.subplots(
        len(test_concepts),
        len(cardinalities),
        figsize=(16 / 2, 9),
        gridspec_kw={"hspace": 0.8},
    )
    for i, concept in enumerate(test_concepts):
        concept_idx = concepts.index(concept)

        for j, cardinality in enumerate(cardinalities):
            subset_idx = _sample_subset(concepts, concept_idx, cardinality)

            _, null_h = model.sample(z, subset_idx, m=m)
            _, test_h = model.sample(z, subset_idx + [concept_idx], m=m)

            null_y = classifier(null_h)
            test_y = classifier(test_h)
            min_y, max_y = np.amin([null_y, test_y]), np.amax([null_y, test_y])

            null_kde = gaussian_kde(null_y)
            test_kde = gaussian_kde(test_y)

            yy = np.linspace(min_y, max_y, 1000)
            null_pdf = null_kde.pdf(yy)
            test_pdf = test_kde.pdf(yy)
            kl = entropy(test_pdf, qk=null_pdf)

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
            ax.set_xlim(0.05, 0.45)
            ax.set_xticks([0.10, 0.20, 0.30, 0.40])
            ax.set_ylim(0, 16)
            if j == 0:
                ax.set_ylabel("Density")
                ax.set_yticks([0, 5, 10, 15])
            else:
                ax.set_ylabel("")
                ax.set_yticks([])
            ax.set_title(
                "\n".join(
                    [
                        f"{concept}",
                        f"cardinality = {cardinality}",
                        # f"KL = {kl:.2f}",
                    ]
                )
            )
            ax.legend()

    plt.savefig(
        os.path.join(figure_dir, f"local_dist_{model.scale}.pdf"), bbox_inches="tight"
    )
    plt.savefig(
        os.path.join(figure_dir, f"local_dist_{model.scale}.png"), bbox_inches="tight"
    )
    plt.show()
    plt.close()
