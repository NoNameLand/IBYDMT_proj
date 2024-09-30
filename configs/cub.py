import os

from ibydmt.utils.config import Config, register_config


@register_config(name="cub")
class CUBConfig(Config):
    def __init__(self):
        super().__init__()
        self.name = os.path.basename(__file__).replace(".py", "")

        data = self.data
        data.dataset = "cub"
        data.backbone = [
            "clip:RN50",
            "clip:ViT-B/32",
            "clip:ViT-L/14",
            "open_clip:ViT-B-32",
            "open_clip:ViT-L-14",
            "flava",
            "align",
            "blip",
        ]
        data.bottleneck = "cub_attribute"
        data.classifier = "zeroshot"
        data.sampler = "attribute"
        data.num_concepts = 14

        testing = self.testing
        testing.significance_level = 0.05
        testing.wealth = "ons"
        testing.bet = "tanh"
        testing.kernel = "rbf"
        testing.kernel_scale_method = "quantile"
        testing.kernel_scale = 0.5
        testing.tau_max = 200
        testing.images_per_class = 10
        testing.cardinality = [1, 2, 4]
        testing.r = 100
