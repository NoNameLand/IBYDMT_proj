import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import find_classes, make_dataset
from torchvision.transforms.functional import to_tensor

from ibydmt.utils.data import register_dataset

workdir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(workdir, "data")


class CountingDataset(Dataset):
    def __init__(self, train):
        super().__init__()
        op = "train" if train else "test"
        op_dir = os.path.join(data_dir, "counting", op)

        self.train = train

        self.image_dir = os.path.join(op_dir, "images")
        self.digits = np.load(os.path.join(op_dir, "digits.npy")).astype(np.float32)

    def __len__(self):
        return self.digits.shape[0]

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, f"{idx}.jpg")
        image = Image.open(image_path)
        return to_tensor(image), self.digits[idx]
