import os

import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import find_classes, make_dataset

from ibydmt.utils.concepts import register_class_concept_trainer
from ibydmt.utils.config import Config
from ibydmt.utils.config import Constants as c
from ibydmt.utils.data import register_dataset

rng = np.random.default_rng()


@register_dataset(name="awa2")
class AwA2(VisionDataset):
    def __init__(self, root, train=None, transform=None):
        super().__init__(root, transform=transform)
        self.op = "train" if train else "test"

        image_root = os.path.join(root, "AwA2", self.op)
        wnids, wnid_to_idx = find_classes(image_root)
        self.classes = [" ".join(wnid.split("+")) for wnid in wnids]

        self.samples = make_dataset(image_root, wnid_to_idx, extensions=".jpg")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


@register_class_concept_trainer(name="awa2")
def train_class_concepts(
    config: Config, concept_class_name: str, workdir=c.WORKDIR, device=c.DEVICE
):
    awa2_dir = os.path.join(workdir, "data", "AwA2")
    classes_path = os.path.join(awa2_dir, "classes.txt")
    predicate_path = os.path.join(awa2_dir, "predicates.txt")
    predicate_continuous_path = os.path.join(
        awa2_dir, "predicate-matrix-continuous.txt"
    )

    with open(classes_path, "r") as f:
        classes = [line.strip().split()[1].replace("+", " ") for line in f]

    with open(predicate_path, "r") as f:
        predicate = [line.strip().split()[1].replace("+", " ") for line in f]

    with open(predicate_continuous_path, "r") as f:
        predicate_continuous = np.loadtxt(f)

    class_idx = classes.index(concept_class_name)
    class_continuous = predicate_continuous[class_idx]

    class_continuous_sorted_idx = np.argsort(class_continuous)[::-1]
    class_sorted_predicate = [predicate[idx] for idx in class_continuous_sorted_idx]
    (class_negative_idx,) = np.where(class_continuous == 0)
    class_negative_predicate = [predicate[idx] for idx in class_negative_idx]

    n = config.data.num_concepts // 2
    positive_predicate = class_sorted_predicate[:n]
    negative_predicate = (
        rng.choice(class_negative_predicate, n, replace=False).squeeze().tolist()
    )
    return positive_predicate + negative_predicate
