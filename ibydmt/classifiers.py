import logging
import os
import pickle
from abc import abstractmethod
from typing import Mapping, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import softmax

from ibydmt.multimodal import get_model, get_text_encoder
from ibydmt.utils.config import Config
from ibydmt.utils.config import Constants as c
from ibydmt.utils.data import get_dataset, get_embedded_dataset

logger = logging.getLogger(__name__)


class Classifier(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.classes = get_dataset(config).classes

    @staticmethod
    def from_pretrained(config: Config, workdir=c.WORKDIR, device=c.DEVICE):
        model = get_classifier(config)
        state_path = model.state_path(workdir=workdir)
        if not os.path.exists(state_path):
            model.train_classifier(device=device)
            model.save(workdir=workdir)
        model.load_state_dict(torch.load(state_path, map_location=device))
        return model.eval()

    @abstractmethod
    def prediction_path(self, workdir=c.WORKDIR):
        pass

    @abstractmethod
    def state_path(self, workdir=c.WORKDIR):
        pass

    @abstractmethod
    def evaluate(self, workdir=c.WORKDIR):
        pass

    @abstractmethod
    def train_classifier(self, device=c.DEVICE):
        pass

    def save(self, workdir=c.WORKDIR):
        os.makedirs(os.path.dirname(self.state_path(workdir=workdir)), exist_ok=True)
        torch.save(self.state_dict(), self.state_path(workdir=workdir))


class ImageClassifier(Classifier):
    def __init__(self, config: Config):
        super().__init__(config)

    def prediction_path(self, workdir=c.WORKDIR):
        prediction_dir = os.path.join(workdir, "results", self.config.name.lower())
        return os.path.join(
            prediction_dir, f"{self.config.data.classifier}_predictions.csv"
        )


class EmbeddingClassifier(Classifier):
    def __init__(self, config: Config):
        super().__init__(config)
        self.embed_dim = get_embedded_dataset(config).embedding.shape[1]

    def prediction_path(self, workdir=c.WORKDIR):
        prediction_dir = os.path.join(workdir, "results", self.config.name.lower())
        return os.path.join(
            prediction_dir,
            f"{self.config.backbone_name()}_{self.config.data.classifier}_predictions.csv",
        )

    def state_path(self, workdir=c.WORKDIR):
        weights_dir = os.path.join(workdir, "weights", self.config.name.lower())
        return os.path.join(
            weights_dir,
            f"{self.config.backbone_name()}_{self.config.data.classifier}.pt",
        )

    @torch.no_grad()
    def evaluate(self, workdir=c.WORKDIR):
        dataset = get_embedded_dataset(self.config, train=False, workdir=workdir)

        output = self(torch.tensor(dataset.embedding)).numpy()
        label = dataset.label

        prediction = np.argmax(output, axis=-1)
        accuracy = np.mean((prediction == label).astype(float))
        logger.info(f"Accuracy: {accuracy:.2%}")

        df = pd.DataFrame(output, columns=dataset.classes)
        prediction_path = self.prediction_path(workdir=workdir)
        os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
        df.to_csv(prediction_path, index=True)


classifier: Mapping[str, Classifier] = {}


def register_classifier(name):
    def register(cls: Classifier):
        if name in classifier:
            raise ValueError(f"Classifier {name} is already registered")
        classifier[name] = cls

    return register


def get_classifier(config: Config) -> Classifier:
    return classifier[config.data.classifier](config)


def get_predictions(config: Config, workdir=c.WORKDIR):
    model = Classifier.from_pretrained(config, workdir)
    prediction_path = model.prediction_path(workdir=workdir)
    if not os.path.exists(prediction_path):
        model.evaluate(workdir)
    return pd.read_csv(prediction_path)


@register_classifier(name="zeroshot")
class ZeroShotClassifier(EmbeddingClassifier):
    def __init__(self, config: Config):
        super().__init__(config)

        self.register_buffer("cbl", torch.zeros(len(self.classes), self.embed_dim))

    def forward(self, h):
        return h @ self.cbl.T

    @torch.no_grad()
    def train_classifier(self, device=c.DEVICE):
        logger.info(
            "Training zero-shot classifier for dataset"
            f" {self.config.data.dataset.lower()} with backbone ="
            f" {self.config.data.backbone}"
        )
        encode_text = get_text_encoder(self.config, device=device)

        dataset = get_dataset(self.config)
        classes = dataset.classes
        prompts = [f"A photo of a {class_name}" for class_name in classes]

        cbl = encode_text(prompts).float()
        cbl = cbl / torch.linalg.norm(cbl, dim=1, keepdim=True)
        cbl = cbl.cpu()
        self.cbl = cbl


@register_classifier(name="mlp")
class MLPClassifier(EmbeddingClassifier):
    BOTTLENECK_DIM = 64

    def __init__(self, config: Config, bottleneck_dim: Optional[int] = None):
        super().__init__(config)
        self.bottleneck_dim = bottleneck_dim or self.BOTTLENECK_DIM
        self.features = nn.Sequential(
            nn.Linear(self.embed_dim, self.bottleneck_dim), nn.ReLU()
        )
        self.output = nn.Linear(self.bottleneck_dim, len(self.classes))

    def forward(self, h):
        h = self.features(h)
        return self.output(h)
