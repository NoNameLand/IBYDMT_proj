import os

from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import find_classes, make_dataset

from ibydmt.utils import splice
from ibydmt.utils.concepts import (
    register_class_concept_trainer,
    register_dataset_concept_trainer,
    register_image_concept_trainer,
)
from ibydmt.utils.config import Config
from ibydmt.utils.config import Constants as c
from ibydmt.utils.data import get_dataset, register_dataset


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

        self.op = "train" if train else "test"
        self.image_root = os.path.join(root, "imagenette2", self.op)

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


@register_dataset_concept_trainer(name="imagenette")
def train_dataset_concepts(config: Config, workdir=c.WORKDIR, device=c.DEVICE):
    return splice.train_dataset_concepts(config, workdir=workdir, device=device)


@register_class_concept_trainer(name="imagenette")
def train_class_concepts(
    config: Config, concept_class_name: str, workdir=c.WORKDIR, device=c.DEVICE
):
    return splice.train_class_concepts(
        config, concept_class_name, workdir=workdir, device=device
    )


@register_image_concept_trainer(name="imagenette")
def train_image_concepts(
    config: Config, concept_image_idx: int, workdir=c.WORKDIR, device=c.DEVICE
):
    dataset = get_dataset(config, workdir=workdir, train=False)
    _, label = dataset[concept_image_idx]
    extra_concepts = {
        "tench": ["trumpet", "selling", "brass", "dispener"],
        "English springer": ["fore", "cathedral", "trumpet", "fishing"],
        "cassette player": ["cathedral", "band", "jazz", "airshow"],
        "chainsaw": ["flew", "trumpet", "obsolete", "airshow"],
        "church": ["trumpet", "fore", "fish", "fishing"],
        "French horn": ["bass", "fish", "cathedral", "obsolete"],
        "garbage truck": ["jazz", "fishing", "obsolete", "fore"],
        "gas pump": ["fishing", "exterior", "putting", "jazz"],
        "golf ball": ["fish", "instrument", "battered", "jazz"],
        "parachute": ["band", "fishing", "instrument", "cathedral"],
    }
    splice_concepts = splice.train_image_concepts(
        config, concept_image_idx, workdir=workdir, device=device
    )
    return splice_concepts[:10] + extra_concepts[dataset.classes[label]]
