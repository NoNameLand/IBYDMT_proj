import os

import ml_collections

from configs.utils import register_config


@register_config(name="imagenette")
def get_config():
    config = ml_collections.ConfigDict()
    config.name = os.path.basename(__file__).replace(".py", "")

    data = config.data = ml_collections.ConfigDict()
    data.dataset = "imagenette"
    data.clip_backbone = "ViT-B/32"
    data.num_concepts = 20

    splice = config.splice = ml_collections.ConfigDict()
    splice.vocab = "mscoco"
    splice.vocab_size = int(1e04)
    splice.l1_penalty = 0.20

    pcbm = config.pcbm = ml_collections.ConfigDict()
    pcbm.alpha = 1e-05
    pcbm.l1_ratio = 0.99

    ckde = config.ckde = ml_collections.ConfigDict()
    ckde.metric = "euclidean"
    ckde.scale_method = "neff"
    ckde.scale = [1000, 2000, 4000]

    testing = config.testing = ml_collections.ConfigDict()
    testing.significance_level = 0.05
    testing.wealth = "ons"
    testing.bet = "tanh"
    testing.kernel = "rbf"
    testing.kernel_scale_method = "quantile"
    testing.kernel_scale = [0.5, 0.7, 0.9]
    testing.tau_max = [100, 200, 400, 800, 1600]
    testing.r = 100
    return config
