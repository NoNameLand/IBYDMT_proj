import logging
import os
import pickle

import clip
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

from ibydmt.utils.concept_data import get_dataset, get_dataset_with_concepts
from ibydmt.utils.config import Config
from ibydmt.utils.config import IBYDMTConstants as c

logger = logging.getLogger(__name__)


class CLIPClassifier:
    def __init__(self, config: Config):
        self.config = config

        self.classes = None
        self.logit_scale = None
        self.classifier = None

    def state_path(self, workdir=c.WORKDIR):
        state_dir = os.path.join(workdir, "weights", self.config.name.lower())
        os.makedirs(state_dir, exist_ok=True)
        return os.path.join(state_dir, f"clip_classifier.pkl")

    @staticmethod
    def load_or_train(config: Config, workdir=c.WORKDIR, device=c.DEVICE):
        model = CLIPClassifier(config)
        state_path = model.state_path(workdir)

        classifier_exists = os.path.exists(state_path)
        if classifier_exists:
            with open(model.state_path(workdir), "rb") as f:
                classes, logit_scale, classifier = pickle.load(f)

            model.classes = classes
            model.logit_scale = logit_scale
            model.classifier = classifier
        else:
            model.train(device=device)
            model.save(workdir=workdir)
        return model

    @staticmethod
    def get_predictions(config: Config, workdir=c.WORKDIR):
        prediction_path = os.path.join(
            workdir, "results", config.name.lower(), "predictions.csv"
        )
        if not os.path.exists(prediction_path):
            model = CLIPClassifier.load_or_train(config, workdir)
            model.predict(workdir)

        return pd.read_csv(prediction_path)

    def save(self, workdir=c.WORKDIR):
        with open(self.state_path(workdir=workdir), "wb") as f:
            pickle.dump((self.classes, self.logit_scale, self.classifier), f)

    def __call__(self, h):
        return h @ self.classifier.T

    @torch.no_grad()
    def train(self, device=c.DEVICE):
        logger.info(
            f"Training CLIP classifier for dataset {self.config.data.dataset.lower()}"
        )
        model, _ = clip.load(self.config.data.clip_backbone, device=device)
        logit_scale = model.logit_scale.cpu().numpy()

        dataset = get_dataset(self.config)
        classes = dataset.classes
        prompts = [f"A photo of a {class_name}" for class_name in classes]
        texts = clip.tokenize(prompts).to(device)

        classifier = model.encode_text(texts).float()
        classifier = classifier / torch.linalg.norm(classifier, dim=1, keepdim=True)
        classifier = classifier.cpu().numpy()

        self.classes = classes
        self.logit_scale = logit_scale
        self.classifier = classifier

    def predict(self, workdir=c.WORKDIR):
        logger.info(
            "Predicting with CLIP classifier on dataset"
            f" {self.config.data.dataset.lower()}"
        )
        dataset = get_dataset_with_concepts(self.config, workdir=workdir, train=False)
        assert dataset.classes == self.classes

        embedding = dataset.embedding
        output = self(embedding)
        probs = softmax(np.exp(self.logit_scale) * output, axis=-1)
        prediction = np.argmax(probs, axis=-1)
        accuracy = np.mean((prediction == dataset.label).astype(float))
        logger.info(f"Accuracy: {accuracy:.2%}")

        results_dir = os.path.join(workdir, "results", self.config.name.lower())
        os.makedirs(results_dir, exist_ok=True)
        df = pd.DataFrame(output, columns=self.classes)
        df.to_csv(os.path.join(results_dir, "predictions.csv"), index=True)
