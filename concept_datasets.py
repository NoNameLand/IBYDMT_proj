import os
import pickle

import clip
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from concept_lib import get_concepts
from datasets import get_dataset, workdir
from models.clip_cbm import CLIPConceptBottleneck


def get_concept_dataset(
    config,
    train=True,
    transform=None,
    return_image=False,
    concept_class_name=None,
    concept_image_idx=None,
):
    return DatasetWithConcepts(
        config,
        train=train,
        transform=transform,
        return_image=return_image,
        concept_class_name=concept_class_name,
        concept_image_idx=concept_image_idx,
    )


class DatasetWithConcepts(Dataset):
    def __init__(
        self,
        config,
        train=True,
        transform=None,
        return_image=False,
        concept_class_name=None,
        concept_image_idx=None,
    ):
        super().__init__()
        self.return_image = return_image
        self.dataset = dataset = get_dataset(config, train=train, transform=transform)

        self.root = dataset.root
        self.train = dataset.train
        self.classes = dataset.classes
        self.concept_name, self.concepts = get_concepts(
            config,
            concept_class_name=concept_class_name,
            concept_image_idx=concept_image_idx,
        )

        op = "train" if train else "val"

        data_dir = os.path.join(self.root, "concept_data")
        os.makedirs(data_dir, exist_ok=True)

        self.data_path = data_path = os.path.join(
            data_dir, f"{op}_{self.concept_name}.pkl"
        )
        if not os.path.exists(data_path):
            self._project_dataset(
                config,
                concept_class_name=concept_class_name,
                concept_image_idx=concept_image_idx,
            )

        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

        self.H, self.Z, self.Y = map(np.array, zip(*self.data))

    def to_df(self):
        data = [[self.classes[y]] + z.tolist() for _, z, y in self.data]
        columns = ["class"] + [concept for concept in self.concepts]
        return pd.DataFrame(data, columns=columns)

    @torch.no_grad()
    def _project_dataset(
        self,
        config,
        concept_class_name=None,
        concept_image_idx=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        print(
            f"Encoding dataset {config.data.dataset} (train = {self.train}) with"
            f" concept bottleneck model (concept_name = {self.concept_name})"
        )
        concept_bottleneck = CLIPConceptBottleneck.load_or_train(
            config,
            workdir,
            concept_class_name=concept_class_name,
            concept_image_idx=concept_image_idx,
        )
        model, preprocess = clip.load(config.data.clip_backbone, device=device)

        dataset = get_dataset(config, train=self.train, transform=preprocess)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

        data = []
        for _data in tqdm(dataloader):
            image, label = _data

            image = image.to(device)

            h = model.encode_image(image).float()
            h = h / torch.linalg.norm(h, dim=1, keepdim=True)

            h = h.cpu().numpy()
            label = label.numpy()
            z = concept_bottleneck(h)

            data.extend(list(zip(h, z, label)))

        with open(self.data_path, "wb") as f:
            pickle.dump(data, f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        h, z, label = self.data[idx]
        if self.return_image:
            image, _ = self.dataset[idx]
            return image, h, z, label
        else:
            return h, z, label
