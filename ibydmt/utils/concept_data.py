import logging
import os

import clip
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ibydmt.utils.concepts import get_concepts
from ibydmt.utils.constants import device, workdir
from ibydmt.utils.data import get_dataset
from ibydmt.utils.models.clip_cbm import CLIPConceptBottleneck

logger = logging.getLogger(__name__)


def get_dataset_with_concepts(
    config,
    workdir=workdir,
    train=True,
    transform=None,
    concept_class_name=None,
    concept_image_idx=None,
    return_image=False,
):
    return DatasetWithConcepts(
        config,
        workdir=workdir,
        train=train,
        transform=transform,
        return_image=return_image,
        concept_class_name=concept_class_name,
        concept_image_idx=concept_image_idx,
    )


@torch.no_grad()
def project_dataset_with_concepts(
    config,
    workdir=workdir,
    train=True,
    concept_class_name=None,
    concept_image_idx=None,
    device=device,
):
    concept_bottleneck = CLIPConceptBottleneck.load_or_train(
        config,
        workdir=workdir,
        concept_class_name=concept_class_name,
        concept_image_idx=concept_image_idx,
    )
    model, preprocess = clip.load(config.data.clip_backbone, device=device)

    dataset = get_dataset(config, workdir=workdir, train=train, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    embedding, semantics, label = [], [], []
    for batch in tqdm(dataloader):
        image, target = batch

        image = image.to(device)

        h = model.encode_image(image).float()
        h = h / torch.linalg.norm(h, dim=1, keepdim=True)
        h = h.cpu().numpy()
        z = concept_bottleneck(h)

        h = h.tolist()
        z = z.tolist()
        target = target.numpy().tolist()

        embedding.extend(h)
        semantics.extend(z)
        label.extend(label)

    return embedding, semantics, label


class DatasetWithConcepts(Dataset):
    def __init__(
        self,
        config,
        workdir=workdir,
        train=True,
        transform=None,
        concept_class_name=None,
        concept_image_idx=None,
        return_image=False,
    ):
        super().__init__()
        self.dataset = dataset = get_dataset(
            config, workdir=workdir, train=train, transform=transform
        )
        self.root = os.path.join(workdir, "concept_data")
        self.train = dataset.train
        self.classes = dataset.classes
        self.concept_name, self.concepts = get_concepts(
            config,
            workdir=workdir,
            concept_class_name=concept_class_name,
            concept_image_idx=concept_image_idx,
        )
        self.return_image = return_image

        op = "train" if train else "test"
        data_dir = os.path.join(self.root, config.data.dataset.lower())
        data_path = os.path.join(data_dir, f"{op}_{self.concept_name}")
        if not os.path.exists(data_path):
            os.makedirs(data_dir, exist_ok=True)

            embedding, semantics, label = project_dataset_with_concepts(
                config,
                workdir=workdir,
                train=train,
                concept_class_name=concept_class_name,
                concept_image_idx=concept_image_idx,
                device=device,
            )
            df = pd.DataFrame(
                {"embedding": embedding, "semantics": semantics, "label": label}
            )
            df.to_parquet(data_path)

        self.data = pd.read_parquet(data_path)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            raise NotImplementedError
