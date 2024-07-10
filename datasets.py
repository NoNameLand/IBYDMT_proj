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


# def get_dataset(config, train=True, transform=None):
#     if config.data.dataset.lower() == "imagenette":
#         root = os.path.join(data_dir, "imagenette2")
#         dataset = Imagenette(root=root, train=train, transform=transform)
#     else:
#         raise NotImplementedError(f"Unknown dataset: {dataset}.")
#     return dataset


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


@register_dataset(name="imagenette")
class Imagenette(VisionDataset):
    WNID_TO_CLASS = {
        "n01440764": ("tench", "Tinca tinca"),
        "n02102040": ("English springer", "English springer spaniel"),
        "n02979186": ("cassette player",),
        "n03000684": ("chainsaw", "chain saw"),
        "n03028079": ("church", "church building"),
        "n03394916": ("French horn", "horn"),
        "n03417042": ("garbage truck", "dustcart"),
        "n03425413": ("gas pump", "gasoline pump", "petrol pump", "island dispenser"),
        "n03445777": ("golf ball",),
        "n03888257": ("parachute", "chute"),
    }

    def __init__(self, root, train=True, transform=None):
        super().__init__(root, transform=transform)
        self.train = train

        self.split = "train" if train else "val"
        self.image_root = os.path.join(root, "imagenette2", self.split)

        self.wnids, self.wnid_to_idx = find_classes(self.image_root)
        self.classes = [self.WNID_TO_CLASS[wnid][0] for wnid in self.wnids]

        self.samples = make_dataset(
            self.image_root, self.wnid_to_idx, extensions=".jpeg"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
