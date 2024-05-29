import os

import ml_collections

from configs.utils import register_config


@register_config(name="gaussian")
def get_config():
    config = ml_collections.ConfigDict()
    config.name = os.path.basename(__file__).replace(".py", "")

    testing = config.testing = ml_collections.ConfigDict()
    testing.significance_level = 0.05
    testing.wealth = "ons"
    testing.bet = "tanh"
    testing.kernel = None
    testing.tau_max = None
    return config
