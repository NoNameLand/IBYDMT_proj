from abc import abstractmethod
from typing import Mapping, Optional

from ibydmt.utils.config import Config
from ibydmt.utils.config import Constants as c


class VisionLanguageModel:
    def __init__(self, backbone: Optional[str] = None, device=c.DEVICE):
        pass

    @abstractmethod
    def encode_text(self, text):
        pass

    @abstractmethod
    def encode_image(self, image):
        pass


models: Mapping[str, VisionLanguageModel] = {}


def register_model(name):
    def register(cls: VisionLanguageModel):
        if name in models:
            raise ValueError(f"Model {name} is already registered")
        models[name] = cls

    return register


def get_model_name_and_backbone(config: Config):
    backbone = config.data.backbone.split(":")
    if len(backbone) == 1:
        backbone.append(None)
    return backbone


def get_model(config: Config, device=c.DEVICE) -> VisionLanguageModel:
    model_name, backbone = get_model_name_and_backbone(config)
    return models[model_name](backbone, device=device)


def get_text_encoder(config: Config, device=c.DEVICE):
    model = get_model(config, device=device)
    return model.encode_text


def get_image_encoder(config: Config, device=c.DEVICE):
    model = get_model(config, device=device)
    return model.encode_image
