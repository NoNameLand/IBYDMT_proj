import functools
import os

import ml_collections
import numpy as np
import pandas as pd
import torch
import wandb
from joblib import Parallel, delayed
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms.functional import to_pil_image, to_tensor
from tqdm import tqdm

from datasets import CountingDataset
from ibydmt import SKIT, cSKIT, xSKIT
from models.counting import CountingNet
from test_lib import tqdm_joblib

DEP_DIGITS = {
    "blue zeros": (0, (31, 119, 180)),
    "orange threes": (3, (255, 127, 14)),
    "green fives": (5, (44, 160, 44)),
    "red threes": (3, (214, 39, 40)),
}
INDEP_DIGITS = {
    "blue twos": (2, (31, 119, 180)),
    "purple sevens": (7, (148, 103, 189)),
}
DIGIT_NAMES = list(DEP_DIGITS.keys()) + list(INDEP_DIGITS.keys())

target_idx = 3
prior, alpha = 0.5, 0.8
concept_idx = list(set(range(len(DIGIT_NAMES))) - {target_idx})

CLASS_NAME = DIGIT_NAMES[target_idx]
CONCEPTS = [DIGIT_NAMES[idx] for idx in concept_idx]

rng = np.random.default_rng()

background_patch = torch.ones(1, 3, 28, 28)

config = ml_collections.ConfigDict()
config.name = "counting"
config.testing = testing = ml_collections.ConfigDict()
testing.significance_level = 0.05
testing.wealth = "ons"
testing.bet = "tanh"
testing.kernel = None
testing.tau_max = None
testing.r = 100

data_dir = os.path.join(os.path.dirname(__file__), "data")
counting_dir = os.path.join(data_dir, "counting")
os.makedirs(counting_dir, exist_ok=True)


def _get_digit_idx(dataset, digit):
    return [idx for idx, (_, y) in enumerate(dataset) if y == digit]


mnist_datasets = {
    "train": MNIST(root=data_dir, train=True, download=True),
    "test": MNIST(root=data_dir, train=False, download=True),
}
mnist_datasets = {
    op: (
        dataset,
        {
            digit: _get_digit_idx(dataset, digit)
            for digit, _ in list(DEP_DIGITS.values()) + list(INDEP_DIGITS.values())
        },
    )
    for op, dataset in mnist_datasets.items()
}


def _image_to_color_tensor(image, color):
    image = Image.fromarray(image.numpy(), mode="L").convert("RGB")
    image = to_tensor(image)

    color = torch.tensor(color)
    color_image = color[:, None, None] / 255 * image
    return color_image + (1 - image)


def _idx_to_color_tensors(mnist_dataset, color, idx):
    return torch.stack(
        [_image_to_color_tensor(mnist_dataset.data[i], color) for i in idx]
    )


def _sample_indep_digit(n):
    value = rng.integers(1, 2, endpoint=True, size=n)
    eps = rng.uniform(-0.5, 0.5, size=n)
    return value + eps


def _sample_digit(n):
    value = rng.integers(0, 2, endpoint=True, size=n)
    eps = rng.uniform(-0.5, 0.5, size=n)
    return value + eps


def _sample_third_digit(first_digit):
    pval = np.zeros((len(first_digit), 3))
    g = np.round(first_digit)
    pval[g == 0] = [3 / 4, 1 / 8, 1 / 8]
    pval[g == 1] = [1 / 8, 3 / 4, 1 / 8]
    pval[g == 2] = [1 / 8, 1 / 8, 3 / 4]
    value = 1 + rng.multinomial(1, pval).argmax(axis=-1)
    eps = rng.uniform(-0.5, 0.5, size=len(first_digit))
    return value + eps


def _sample_target_digit(second_digit, third_digit):
    assert len(second_digit) == len(third_digit)

    g = np.round(second_digit) * np.round(third_digit)
    p = np.zeros((len(second_digit)))

    t, alpha = 3, 0.9
    p[g >= t] = alpha
    p[g < t] = 1 - alpha

    outcome = rng.binomial(1, p)
    eps = rng.uniform(-0.5, 0.5, size=len(second_digit))
    return 2 + outcome + eps


def _sample_dep_digits(n):
    first_digit = _sample_digit(n)
    second_digit = _sample_digit(n)
    third_digit = _sample_third_digit(first_digit)
    target_digit = _sample_target_digit(second_digit, third_digit)
    return np.stack([first_digit, second_digit, third_digit, target_digit], axis=1)


def sample_digits(n):
    dep_digits = _sample_dep_digits(n)
    indep_digits = np.stack(
        [_sample_indep_digit(n) for _ in range(len(INDEP_DIGITS))], axis=1
    )
    return dep_digits, indep_digits


def sample_image(op, dep_digits, indep_digits):
    l = 4

    mnist_dataset, digit_idx = mnist_datasets[op]

    total_digits = dep_digits.sum() + indep_digits.sum()
    n_background_patches = l**2 - total_digits

    dep_digit_patches = torch.cat(
        [
            _idx_to_color_tensors(
                mnist_dataset,
                color,
                rng.choice(digit_idx[digit], size=dep_digits[i], replace=False),
            )
            for i, (digit, color) in enumerate(DEP_DIGITS.values())
            if dep_digits[i] > 0
        ]
    )
    indep_digit_patches = torch.cat(
        [
            _idx_to_color_tensors(
                mnist_dataset,
                color,
                rng.choice(digit_idx[digit], size=indep_digits[i], replace=False),
            )
            for i, (digit, color) in enumerate(INDEP_DIGITS.values())
            if indep_digits[i] > 0
        ]
    )
    background_patches = background_patch.expand(n_background_patches, -1, -1, -1)

    patches = torch.cat([dep_digit_patches, indep_digit_patches, background_patches])
    patches = patches[rng.permutation(l**2)]
    patches = patches.view(l, l, 3, 28, 28)
    patches = patches.permute(2, 0, 3, 1, 4).contiguous()
    return patches.view(3, 28 * l, 28 * l)


def _sample_cond_digits(digits, cond_idx):
    n, d = digits.shape

    if len(cond_idx) == 0:
        cond_digits = sample_digits(n)
    if len(cond_idx) == d:
        cond_digits = digits
    else:
        cond_digits = np.empty((n, d))

        fn_list = [
            (1, _sample_digit),
            (3, _sample_indep_digit),
            (4, _sample_indep_digit),
        ]
        for idx, fn in fn_list:
            cond_digits[:, idx] = digits[:, idx] if idx in cond_idx else fn(n)

        if 2 in cond_idx:
            cond_digits[:, 2] = third_digit = digits[:, 2]

            prior = 1 / 3 * np.ones((3, 3))
            likelihood = np.array(
                [
                    [3 / 4, 1 / 8, 1 / 8],
                    [1 / 8, 3 / 4, 1 / 8],
                    [1 / 8, 1 / 8, 3 / 4],
                ]
            )
            posterior = prior * likelihood
            posterior /= posterior.sum(axis=1, keepdims=True)

            pvals = posterior[np.round(third_digit).astype(int) - 1]
            eps = rng.uniform(-0.5, 0.5, size=n)
            value = rng.multinomial(1, pvals).argmax(axis=-1)
            cond_digits[:, 0] = value + eps
        else:
            cond_digits[:, 0] = first_digit = (
                digits[:, 0] if 0 in cond_idx else _sample_digit(n)
            )
            cond_digits[:, 2] = _sample_third_digit(first_digit)
    return cond_digits


def cond_p(digits, cond_idx, m=1, return_target_digit=False, do_sample_image=False):
    if digits.ndim == 1:
        digits.reshape(1, -1)
        digits = np.tile(digits, (m, 1))

    cond_digits = _sample_cond_digits(digits, cond_idx)

    dep_digits, indep_digits = cond_digits[:, :3], cond_digits[:, 3:]
    second_digit, third_digit = dep_digits[:, 1], dep_digits[:, 2]
    target_digit = _sample_target_digit(second_digit, third_digit)

    if do_sample_image:
        dep_digits = np.hstack([dep_digits, target_digit[:, None]])

        output = torch.stack(
            [
                sample_image(
                    "test",
                    np.round(dep_digits[i]).astype(int),
                    np.round(indep_digits[i]).astype(int),
                )
                for i in range(len(cond_digits))
            ]
        )
    else:
        output = cond_digits

    if return_target_digit:
        return output, target_digit
    else:
        return output


def generate(train):
    op = "train" if train else "test"

    op_dir = os.path.join(counting_dir, op)
    image_dir = os.path.join(op_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    n = int(5e04) if train else int(1e04)
    digits = np.empty((n, len(DIGIT_NAMES)))

    dep_digits, indep_digits = sample_digits(n)

    for i in tqdm(range(n)):
        image_path = os.path.join(image_dir, f"{i}.jpg")
        image = sample_image(
            op,
            np.round(dep_digits[i]).astype(int),
            np.round(indep_digits[i]).astype(int),
        )
        image = to_pil_image(image)
        image.save(image_path)

    digits = np.hstack([dep_digits, indep_digits])
    np.save(os.path.join(op_dir, "digits.npy"), digits)


def train(workdir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = os.path.join(workdir, "checkpoints", "counting")
    os.makedirs(checkpoint_dir, exist_ok=True)

    datasets = {op: CountingDataset(op == "train") for op in ["train", "test"]}
    loaders = {
        op: DataLoader(d, batch_size=64, shuffle=op == "train", num_workers=4)
        for op, d in datasets.items()
    }

    model = CountingNet(len(DIGIT_NAMES), device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-04, weight_decay=1e-05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    for _ in range(6):
        for op, loader in loaders.items():
            if op == "train":
                model.train()
                torch.set_grad_enabled(True)
            else:
                model.eval()
                torch.set_grad_enabled(False)

            running_loss, running_accuracy, running_samples = 0.0, 0.0, 0
            for i, data in enumerate(tqdm(loader)):
                image, target = data

                image = image.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                loss, accuracy = model.loss_fn(image, target)
                if op == "train":
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                running_accuracy += accuracy.item()
                running_samples += image.size(0)

                log_step = 20
                if op == "train" and (i + 1) % log_step == 0:
                    wandb.log(
                        {
                            "train/loss": running_loss / running_samples,
                            "train/accuracy": running_accuracy / running_samples,
                        }
                    )
                    running_loss, running_accuracy, running_samples = 0.0, 0.0, 0

            if op == "train":
                scheduler.step()

            if op == "test":
                wandb.log(
                    {
                        "test/loss": running_loss / running_samples,
                        "test/accuracy": running_accuracy / running_samples,
                    }
                )
                print(f"Test accuracy: {running_accuracy / running_samples:.2%}")
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "net.pt"))


@torch.no_grad()
def predict(workdir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = os.path.join(workdir, "results", "counting")
    os.makedirs(results_dir, exist_ok=True)

    dataset = CountingDataset(train=False)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    model = CountingNet.from_pretrained(len(DIGIT_NAMES), workdir, device=device)

    predictions = []
    for _, data in enumerate(tqdm(loader)):
        image, _ = data

        image = image.to(device)
        prediction = model(image).cpu().numpy()
        prediction = np.round(prediction) + rng.uniform(
            -0.5, 0.5, size=prediction.shape
        )
        predictions.append(prediction)

    predictions = np.concatenate(predictions)
    predictions = pd.DataFrame(predictions, columns=DIGIT_NAMES)
    predictions.to_csv(os.path.join(results_dir, "predictions.csv"))


def explain(workdir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = os.path.join(workdir, "results", "counting", "explanations")
    os.makedirs(results_dir, exist_ok=True)

    dataset = CountingDataset(train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = CountingNet.from_pretrained(len(DIGIT_NAMES), workdir, device=device)

    cam = GradCAM(model=model, target_layers=[model.resnet.layer4[-1]])
    targets = [ClassifierOutputTarget(target_idx)]

    for idx, data in enumerate(tqdm(loader)):
        image, _ = data

        image = image.to(device)
        explanation = cam(input_tensor=image, targets=targets).squeeze()

        output_path = os.path.join(results_dir, f"{idx}.npy")
        np.save(output_path, explanation)


def _test_preamble(use_model, workdir):
    if use_model:
        results_dir = os.path.join(workdir, "results", "counting")

        predictions = pd.read_csv(os.path.join(results_dir, "predictions.csv"))

        Y, Z = predictions[CLASS_NAME].values, predictions[CONCEPTS].values
    else:
        dataset = CountingDataset(train=False)
        digits = pd.DataFrame(dataset.digits, columns=DIGIT_NAMES)

        Y, Z = digits[CLASS_NAME].values, digits[CONCEPTS].values

    return Y, Z


def _test_global(config, workdir, use_model, **testing_kw):
    testing_config = config.testing
    for key, value in testing_kw.items():
        setattr(testing_config, key, value)

    print(
        f"Testing for global semantic independence with kernel = {testing_config.kernel}, tau_max = {testing_config.tau_max}"
    )

    test_type = "global_model" if use_model else "global"
    test_results_dir = os.path.join(workdir, "results", "counting", test_type)
    os.makedirs(test_results_dir, exist_ok=True)

    Y, Z = _test_preamble(use_model, workdir)

    def test(j, concept):
        Z_concept = Z[:, j]

        rejected_hist, tau_hist = [], []
        for _ in range(config.testing.r):
            pi = np.random.permutation(len(Y))
            pi_Y, pi_Z = Y[pi], Z_concept[pi]

            tester = SKIT(testing_config)
            rejected, tau = tester.test(pi_Y, pi_Z)

            rejected_hist.append(rejected)
            tau_hist.append(tau)

        return {
            "class_name": CLASS_NAME,
            "concept": concept,
            "rejected": rejected_hist,
            "tau": tau_hist,
        }

    with tqdm_joblib(tqdm(desc="Testing", total=len(CONCEPTS))):
        results = Parallel(n_jobs=-1)(
            delayed(test)(j, concept) for j, concept in enumerate(CONCEPTS)
        )

    np.save(
        os.path.join(
            test_results_dir, f"{testing_config.kernel}_{testing_config.tau_max}.npy"
        ),
        results,
        allow_pickle=True,
    )


def _test_global_cond(config, workdir, use_model, **testing_kw):
    testing_config = config.testing
    for key, value in testing_kw.items():
        setattr(testing_config, key, value)

    print(
        f"Testing for global conditional semantic independence with kernel = {testing_config.kernel}, tau_max = {testing_config.tau_max}"
    )

    test_type = "global_cond_model" if use_model else "global_cond"
    test_results_dir = os.path.join(workdir, "results", "counting", test_type)
    os.makedirs(test_results_dir, exist_ok=True)

    Y, Z = _test_preamble(use_model, workdir)

    def test(j, concept):
        rejected_hist, tau_hist = [], []
        for _ in range(config.testing.r):
            pi = np.random.permutation(len(Y))
            pi_Y, pi_Z = Y[pi], Z[pi]

            tester = cSKIT(testing_config)
            rejected, tau = tester.test(pi_Y, pi_Z, j, cond_p)

            rejected_hist.append(rejected)
            tau_hist.append(tau)

        return {
            "class_name": CLASS_NAME,
            "concept": concept,
            "rejected": rejected_hist,
            "tau": tau_hist,
        }

    with tqdm_joblib(tqdm(desc="Testing", total=len(CONCEPTS))):
        results = Parallel(n_jobs=-1)(
            delayed(test)(j, concept) for j, concept in enumerate(CONCEPTS)
        )
        print(results)

    np.save(
        os.path.join(
            test_results_dir, f"{testing_config.kernel}_{testing_config.tau_max}.npy"
        ),
        results,
        allow_pickle=True,
    )


def _test_local_cond(config, workdir, use_model, **testing_kw):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    testing_config = config.testing
    for key, value in testing_kw.items():
        setattr(testing_config, key, value)

    print(
        f"Testing for local conditional semantic independence with kernel = {testing_config.kernel}, tau_max = {testing_config.tau_max}"
    )

    test_type = "local_cond_model" if use_model else "local_cond"
    test_results_dir = os.path.join(workdir, "results", "counting", test_type)
    os.makedirs(test_results_dir, exist_ok=True)

    _, Z = _test_preamble(use_model, workdir)

    _cond_p = functools.partial(cond_p, do_sample_image=True, return_target_digit=True)

    model = CountingNet.from_pretrained(len(DIGIT_NAMES), workdir, device=device)

    @torch.no_grad()
    def _f(input):
        image, target_digit = input

        if use_model:
            image = image.to(device)

            batch_size = 128
            output = torch.cat(
                [
                    model(image[i : i + batch_size])
                    for i in range(0, len(image), batch_size)
                ]
            )
            # output = model(image)
            output = output.cpu().numpy()
            output = output[:, target_idx]
            return np.round(output) + rng.uniform(-0.5, 0.5, size=output.shape)
        else:
            return target_digit

    test_idx, cond_idx = 2, 1
    test_concept = CONCEPTS[test_idx]

    def test(idx, z):
        rejected_hist, tau_hist = [], []

        for _ in range(config.testing.r):
            tester = xSKIT(testing_config)

            rejected, tau = tester.test(z, test_idx, [cond_idx], _cond_p, _f)

            rejected_hist.append(rejected)
            tau_hist.append(tau)

        return {
            "idx": idx,
            "class_name": CLASS_NAME,
            "concept": test_concept,
            "rejected": rejected_hist,
            "tau": tau_hist,
        }

    idx = []
    for n in [2, 1, 0]:
        mask = np.round(Z[:, cond_idx]).astype(int) == n
        mask_idx = np.nonzero(mask)[0]
        idx.extend(mask_idx[:1])
    np.save(os.path.join(test_results_dir, "idx.npy"), idx)
    raise NotImplementedError

    with tqdm_joblib(tqdm(desc=f"Testing", total=len(idx))):
        results = Parallel(n_jobs=10)(delayed(test)(_idx, Z[_idx]) for _idx in idx)

    np.save(
        os.path.join(
            test_results_dir, f"{testing_config.kernel}_{testing_config.tau_max}.npy"
        ),
        results,
        allow_pickle=True,
    )


def test(type, workdir, use_model, **testing_kw):
    if type == "global":
        test_fn = _test_global
    elif type == "global_cond":
        test_fn = _test_global_cond
    elif type == "local_cond":
        test_fn = _test_local_cond
    else:
        raise ValueError(f"Unknown test type: {type}.")
    test_fn(config, workdir, use_model, **testing_kw)
