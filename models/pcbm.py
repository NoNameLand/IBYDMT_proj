import os
import pickle

import numpy as np
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm

from concept_datasets import get_concept_dataset
from concept_lib import get_concepts


class PCBM:
    def __init__(self, config, concept_class_name=None):
        self.config = config

        self.concept_name, self.concepts = get_concepts(
            config, concept_class_name=concept_class_name
        )
        self.concept_class_name = concept_class_name

        pcbm_config = config.get("pcbm", {})
        self.alpha = pcbm_config.get("alpha", 1e-05)
        self.l1_ratio = pcbm_config.get("l1_ratio", 0.99)
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

    def state_path(self, workdir):
        state_dir = os.path.join(workdir, "weights", self.config.name.lower())
        os.makedirs(state_dir, exist_ok=True)
        return os.path.join(state_dir, f"pcbm_{self.concept_name}.pkl")

    @staticmethod
    def load_or_train(config, workdir, concept_class_name=None):
        model = PCBM(config, concept_class_name=concept_class_name)
        state_path = model.state_path(workdir)

        pcbm_exists = os.path.exists(state_path)
        if pcbm_exists:
            with open(state_path, "rb") as f:
                concepts, _ = pickle.load(f)

            if concepts != model.concepts:
                pcbm_exists = False

        if pcbm_exists:
            with open(state_path, "rb") as f:
                _, classifier = pickle.load(f)
            model.classifier_hist = classifier
        else:
            model.train()
            model.eval()
            model.save(workdir)
        return model

    def save(self, workdir):
        with open(self.state_path(workdir), "wb") as f:
            pickle.dump((self.concepts, self.classifier_hist), f)

    def _class_accuracy(self, predictions, Y):
        classes = np.unique(Y)
        class_accuracy = []
        for _class in classes:
            idx = Y == _class
            class_accuracy.append(np.mean((predictions[idx] == Y[idx]).astype(float)))
        return class_accuracy

    def train(self):
        dataset = get_concept_dataset(
            self.config, train=True, concept_class_name=self.concept_class_name
        )

        Z, Y = dataset.Z, dataset.Y

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
            classifier.fit(Z, Y)

            predictions = classifier.predict(Z)
            accuracy = np.mean((predictions == Y).astype(float))
            class_accuracy = self._class_accuracy(predictions, Y)

            accuracy_hist.append(accuracy)
            class_accuracy_hist.append(class_accuracy)
            self.classifier_hist.append(classifier)

        accuracy = np.array(accuracy_hist)
        class_accuracy = np.array(class_accuracy_hist)

        print(
            f"PCBM training accuracy: {accuracy.mean():.2%} (std ="
            f" {accuracy.std():.2%})"
        )
        for class_idx, class_name in enumerate(dataset.classes):
            _class_accuracy = class_accuracy[:, class_idx]
            print(
                f"\t{class_name}: {_class_accuracy.mean():.2%} (std ="
                f" {_class_accuracy.std():.2%})"
            )

    def eval(self):
        dataset = get_concept_dataset(
            self.config, train=False, concept_class_name=self.concept_class_name
        )

        Z, Y = dataset.Z, dataset.Y

        accuracy_hist, class_accuracy_hist = [], []
        for classifier in self.classifier_hist:
            predictions = classifier.predict(Z)

            accuracy = np.mean((predictions == Y).astype(float))
            class_accuracy = self._class_accuracy(predictions, Y)

            accuracy_hist.append(accuracy)
            class_accuracy_hist.append(class_accuracy)

        accuracy = np.array(accuracy_hist)
        class_accuracy = np.array(class_accuracy_hist)

        print(
            f"PCBM validation accuracy: {accuracy.mean():.2%} (std ="
            f" {accuracy.std():.2%})"
        )
        for class_idx, class_name in enumerate(dataset.classes):
            _class_accuracy = class_accuracy[:, class_idx]
            print(
                f"\t{class_name}: {_class_accuracy.mean():.2%} (std ="
                f" {_class_accuracy.std():.2%})"
            )
