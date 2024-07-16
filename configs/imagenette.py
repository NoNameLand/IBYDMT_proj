import os

from ibydmt.utils.config import Config, register_config


@register_config(name="imagenette")
class ImagenetteConfig(Config):
    def __init__(self):
        super().__init__()
        self.name = os.path.basename(__file__).replace(".py", "")

        data = self.data
        data.dataset = "imagenette"
        data.clip_backbone = "ViT-B/32"
        data.num_concepts = 20

        splice = self.splice
        splice.vocab = "mscoco"
        splice.vocab_size = int(1e04)
        splice.l1_penalty = 0.20

        pcbm = self.pcbm
        pcbm.alpha = 1e-05
        pcbm.l1_ratio = 0.99

        ckde = self.ckde
        ckde.metric = "euclidean"
        ckde.scale_method = "neff"
        ckde.scale = [1000, 2000, 4000]

        testing = self.testing
        testing.significance_level = 0.05
        testing.fdr_control = [False, True]
        testing.wealth = "ons"
        testing.bet = "tanh"
        testing.kernel = "rbf"
        testing.kernel_scale_method = "quantile"
        testing.kernel_scale = [0.5, 0.7, 0.9]
        testing.tau_max = [400, 800, 1600]
        testing.r = 100
