import matplotlib.pyplot as plt
import seaborn as sns

from ibydmt.classifiers import ZeroShotClassifier
from ibydmt.samplers import get_sampler
from ibydmt.tester import get_local_test_idx, sample_random_subset
from ibydmt.utils.concept_data import get_dataset_with_concepts
from ibydmt.utils.config import ConceptType, Config
from ibydmt.utils.config import Constants as c
from ibydmt.utils.data import get_dataset


def viz_cond_pdf(config: Config, concept_type: str, workdir=c.WORKDIR):
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
    config: Config, concept_type: str, workdir=c.WORKDIR, device=c.DEVICE
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
