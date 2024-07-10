import os

import torch

workdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
