import logging
import os
import pickle
from test import setup_logging

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm

import configs
import datasets
from ibydmt.tester import sweep
from ibydmt.utils.config import Constants as c
from ibydmt.utils.config import get_config
from ibydmt.utils.data import get_dataset, get_embedded_dataset

logger = logging.getLogger(__name__)

setup_logging("INFO")

workdir = c.WORKDIR
data_dir = os.path.join(workdir, "data")
cub_dir = os.path.join(data_dir, "CUB")

config = get_config("cub")
train_dataset = get_dataset(config, train=True)
test_dataset = get_dataset(config, train=False)

train_image_idx = train_dataset.image_idx
test_image_idx = test_dataset.image_idx

train_df_path = os.path.join(cub_dir, "train_cub_cav.parquet")
test_df_path = os.path.join(cub_dir, "test_cub_cav.parquet")

if not os.path.exists(train_df_path) or not os.path.exists(test_df_path):
    image_attributes_path = os.path.join(
        cub_dir, "attributes", "image_attribute_labels.txt"
    )

    image_idx, attribute_idx, label, split = [], [], [], []
    with open(image_attributes_path, "r") as f:
        for line in tqdm(f):
            chunks = line.strip().split()
            if len(chunks) != 5:
                continue

            _image_idx, _attribute_idx, _label, _confidence, _ = chunks
            if int(_confidence) >= 3:
                _image_idx = int(_image_idx)
                _attribute_idx = int(_attribute_idx)
                _label = int(_label)

                _split = "train" if _image_idx in train_image_idx else "test"

                image_idx.append(_image_idx)
                attribute_idx.append(_attribute_idx)
                label.append(_label)
                split.append(_split)

    df = pd.DataFrame(
        {
            "image_idx": image_idx,
            "attribute_idx": attribute_idx,
            "label": label,
            "split": split,
        }
    )
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    train_df.to_parquet(train_df_path)
    test_df.to_parquet(test_df_path)

train_df = pd.read_parquet(train_df_path)
test_df = pd.read_parquet(test_df_path)

delta = 0.20
positive_ratio = train_df.groupby("attribute_idx")["label"].mean()
attribute_idx = positive_ratio[
    (positive_ratio > delta) & (positive_ratio < (1 - delta))
].index.values

backbone_configs = sweep(config, sweep_keys=["data.backbone"])
for backbone_config in backbone_configs:
    logger.info(f"Training CAVs for {backbone_config.data.backbone}")
    backbone = backbone_config.backbone_name()

    train_embedded_dataset = get_embedded_dataset(backbone_config, train=True)
    test_embedded_dataset = get_embedded_dataset(backbone_config, train=False)

    w = np.zeros((len(attribute_idx), train_embedded_dataset.embedding.shape[1]))
    auc = np.zeros(len(attribute_idx))

    for idx, _attribute_idx in enumerate(attribute_idx):
        _train_df = train_df[train_df["attribute_idx"] == _attribute_idx]
        _test_df = test_df[test_df["attribute_idx"] == _attribute_idx]

        _train_embedding, _train_label = [], []
        for i, row in _train_df.iterrows():
            _image_idx = row["image_idx"]
            _label = row["label"]

            _embedding_idx = train_image_idx.index(_image_idx)
            _train_embedding.append(train_embedded_dataset.embedding[_embedding_idx])
            _train_label.append(_label)

        _test_embedding, _test_label = [], []
        for i, row in _test_df.iterrows():
            _image_idx = int(row["image_idx"])
            _label = int(row["label"])

            _embedding_idx = test_image_idx.index(_image_idx)
            _test_embedding.append(test_embedded_dataset.embedding[_embedding_idx])
            _test_label.append(_label)

        logger.info(f"Training CAV for attribute {_attribute_idx}")

        _train_embedding = np.array(_train_embedding)
        _train_label = np.array(_train_label)

        _test_embedding = np.array(_test_embedding)
        _test_label = np.array(_test_label)

        _lm = SGDClassifier(fit_intercept=False, class_weight="balanced")
        _lm.fit(_train_embedding, _train_label)
        _w = _lm.coef_
        _w /= np.linalg.norm(_w)

        semantics = _test_embedding @ _w.T
        fpr, tpr, thresholds = metrics.roc_curve(_test_label, semantics.flatten())
        _auc = metrics.auc(fpr, tpr)
        logger.info(f"AUC for attribute {_attribute_idx}: {_auc:.2f}")

        w[idx] = _w
        auc[idx] = _auc

    state_path = os.path.join(
        workdir, "weights", "cub", f"{backbone_config.backbone_name()}_cav.pkl"
    )
    with open(state_path, "wb") as f:
        pickle.dump((attribute_idx, w, auc), f)
