import os

from ibydmt.utils.config import Config, register_config


@register_config(name="synthetic")
class SyntheticConfig(Config):
    def __init__(self):
        super().__init__()
        self.name = os.path.basename(__file__).replace(".py", "")

        testing = self.testing
        testing.significance_level = 0.05
        testing.wealth = "ons"
        testing.bet = "tanh"
        testing.kernel = None
        testing.kernel_scale_method = None
        testing.kernel_scale = None
        testing.tau_max = 1000
        testing.r = 100
