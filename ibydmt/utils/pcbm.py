import logging
import os
import pickle
from typing import Optional

import numpy as np
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm

from ibydmt.utils.concept_data import get_dataset_with_concepts
from ibydmt.utils.concepts import get_concepts
from ibydmt.utils.config import Config
from ibydmt.utils.config import Constants as c

logger = logging.getLogger(__name__)


class PCBM:
    def __init__(
        self,
        config: Config,
        workdir=c.WORKDIR,
        concept_class_name: Optional[str] = None,
    ):
        self.config = config

        self.concept_name, self.concepts = get_concepts(
            config, workdir=workdir, concept_class_name=concept_class_name
        )
        self.concept_class_name = concept_class_name

        self.alpha = config.pcbm.alpha
        self.l1_ratio = config.pcbm.l1_ratio
        self.r = config.testing.r

        self.classifier_hist = []

    def weights(self, reduction="mean"):
        weights = np.array([classifier.coef_ for classifier in self.classifier_hist])

        if reduction == "mean":
            weights = weights.mean(axis=0)
        elif reduction == "none":
            pass
        else:
            raise NotImplementedError(f"Reduction method {reduction} not implemented")
        return weights

    def state_path(self, workdir=c.WORKDIR):
        state_dir = os.path.join(workdir, "weights", self.config.name.lower())
        os.makedirs(state_dir, exist_ok=True)
        return os.path.join(
            state_dir, f"{self.config.backbone_name()}_pcbm_{self.concept_name}.pkl"
        )

    @staticmethod
    def load_or_train(
        config, workdir=c.WORKDIR, concept_class_name: Optional[str] = None
    ):
        model = PCBM(config, concept_class_name=concept_class_name)
        state_path = model.state_path(workdir=workdir)

        pcbm_exists = os.path.exists(state_path)
        if pcbm_exists:
            with open(state_path, "rb") as f:
                concepts, _ = pickle.load(f)

            if concepts != model.concepts:
                pcbm_exists = False

        if pcbm_exists:
            with open(state_path, "rb") as f:
                _, classifier_hist = pickle.load(f)
            model.classifier_hist = classifier_hist
        else:
            model.train()
            model.eval()
            model.save(workdir)
        return model

    def save(self, workdir=c.WORKDIR):
        with open(self.state_path(workdir=workdir), "wb") as f:
            pickle.dump((self.concepts, self.classifier_hist), f)

    def class_accuracy(self, prediction, label):
        confusion_matrix = metrics.confusion_matrix(label, prediction)
        return confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)

    def train(self):
        dataset = get_dataset_with_concepts(
            self.config, train=True, concept_class_name=self.concept_class_name
        )

        semantics, label = dataset.semantics, dataset.label

        accuracy_hist, class_accuracy_hist = [], []
        for _ in tqdm(range(self.r)):
            classifier = SGDClassifier(
                loss="log_loss",
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                verbose=0,
                penalty="elasticnet",
                max_iter=int(1e04),
                fit_intercept=False,
            )
            classifier.fit(semantics, label)

            prediction = classifier.predict(semantics)
            accuracy = np.mean((prediction == label).astype(float))
            class_accuracy = self.class_accuracy(prediction, label)

            accuracy_hist.append(accuracy)
            class_accuracy_hist.append(class_accuracy)
            self.classifier_hist.append(classifier)

        accuracy = np.array(accuracy_hist)
        class_accuracy = np.array(class_accuracy_hist)

        logger.info(
            f"PCBM training accuracy: {accuracy.mean():.2%} (std ="
            f" {accuracy.std():.2%})"
        )
        for class_idx, class_name in enumerate(dataset.classes):
            _class_accuracy = class_accuracy[:, class_idx]
            logger.info(
                f"\t{class_name}: {_class_accuracy.mean():.2%} (std ="
                f" {_class_accuracy.std():.2%})"
            )

    def eval(self):
        dataset = get_dataset_with_concepts(
            self.config, train=False, concept_class_name=self.concept_class_name
        )

        semantics, label = dataset.semantics, dataset.label

        accuracy_hist, class_accuracy_hist = [], []
        for classifier in self.classifier_hist:
            predictions = classifier.predict(semantics)

            accuracy = np.mean((predictions == label).astype(float))
            class_accuracy = self.class_accuracy(predictions, label)

            accuracy_hist.append(accuracy)
            class_accuracy_hist.append(class_accuracy)

        accuracy = np.array(accuracy_hist)
        class_accuracy = np.array(class_accuracy_hist)

        logger.info(
            f"PCBM validation accuracy: {accuracy.mean():.2%} (std ="
            f" {accuracy.std():.2%})"
        )
        for class_idx, class_name in enumerate(dataset.classes):
            _class_accuracy = class_accuracy[:, class_idx]
            logger.info(
                f"\t{class_name}: {_class_accuracy.mean():.2%} (std ="
                f" {_class_accuracy.std():.2%})"
            )
