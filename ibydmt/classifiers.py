import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

from ibydmt.utils.config import Config
from ibydmt.utils.config import Constants as c
from ibydmt.utils.data import get_dataset, get_embedded_dataset
from ibydmt.utils.multimodal import get_text_encoder

logger = logging.getLogger(__name__)


class ZeroShotClassifier:
    def __init__(self, config: Config):
        self.config = config

        self.classes = None
        self.classifier = None

    def state_path(self, workdir=c.WORKDIR):
        state_dir = os.path.join(workdir, "weights", self.config.name.lower())
        os.makedirs(state_dir, exist_ok=True)
        return os.path.join(state_dir, f"{self.config.backbone_name()}_classifier.pkl")

    @staticmethod
    def prediction_path(config: Config, workdir=c.WORKDIR):
        prediction_dir = os.path.join(workdir, "results", config.name.lower())
        return os.path.join(prediction_dir, f"{config.backbone_name()}_predictions.csv")

    @staticmethod
    def load_or_train(config: Config, workdir=c.WORKDIR, device=c.DEVICE):
        model = ZeroShotClassifier(config)
        state_path = model.state_path(workdir)

        classifier_exists = os.path.exists(state_path)
        if classifier_exists:
            with open(model.state_path(workdir), "rb") as f:
                classes, classifier = pickle.load(f)

            model.classes = classes
            model.classifier = classifier
        else:
            model.train(device=device)
            model.save(workdir=workdir)
        return model

    @staticmethod
    def get_predictions(config: Config, workdir=c.WORKDIR):
        prediction_path = ZeroShotClassifier.prediction_path(config, workdir)
        if not os.path.exists(prediction_path):
            model = ZeroShotClassifier.load_or_train(config, workdir)
            model.predict(workdir)

        return pd.read_csv(prediction_path)

    def save(self, workdir=c.WORKDIR):
        with open(self.state_path(workdir=workdir), "wb") as f:
            pickle.dump((self.classes, self.classifier), f)

    def __call__(self, h):
        return h @ self.classifier.T

    @torch.no_grad()
    def train(self, device=c.DEVICE):
        logger.info(
            "Training zero-shot classifier for dataset"
            f" {self.config.data.dataset.lower()} with backbone ="
            f" {self.config.data.backbone}"
        )
        encode_text = get_text_encoder(self.config, device=device)

        dataset = get_dataset(self.config)
        classes = dataset.classes
        prompts = [f"A photo of a {class_name}" for class_name in classes]

        classifier = encode_text(prompts).float()
        classifier /= torch.linalg.norm(classifier, dim=1, keepdim=True)
        classifier = classifier.cpu().numpy()

        self.classes = classes
        self.classifier = classifier

    @torch.no_grad()
    def predict(self, workdir=c.WORKDIR):
        logger.info(
            f"Predicting with {self.config.data.backbone} zero-shot classifier on"
            f" dataset {self.config.data.dataset.lower()}"
        )
        dataset = get_embedded_dataset(self.config, train=False, workdir=workdir)
        assert dataset.classes == self.classes

        output = self(dataset.embedding)
        label = dataset.label

        probs = softmax(output, axis=-1)
        prediction = np.argmax(probs, axis=-1)
        accuracy = np.mean((prediction == label).astype(float))
        logger.info(f"Accuracy: {accuracy:.2%}")

        df = pd.DataFrame(output, columns=self.classes)

        prediction_path = self.prediction_path(self.config, workdir)
        os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
        df.to_csv(prediction_path, index=True)
