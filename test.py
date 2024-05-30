import argparse

from configs.utils import get_config
from test_lib import test


def test_type(value):
    if value not in ["all", "global", "global_cond", "local_cond"]:
        raise argparse.ArgumentTypeError(f"Invalid test type: {value}")
    return value


def concept_type(value):
    if value not in ["dataset", "class", "image"]:
        raise argparse.ArgumentTypeError(f"Invalid concept type: {value}")
    return value


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str)
    parser.add_argument("--workdir", type=str, default="./")
    parser.add_argument("--test_type", type=test_type, default="all")
    parser.add_argument("--concept_type", type=concept_type, default="dataset")
    parser.add_argument("--kernel_scale", type=float, default=None)
    parser.add_argument("--ckde_scale", type=float, default=None)
    parser.add_argument("--tau_max", type=int, default=None)
    return parser.parse_args()


def main(args):
    config_name = args.config_name
    workdir = args.workdir
    test_type = args.test_type
    concept_type = args.concept_type
    kernel_scale = args.kernel_scale
    ckde_scale = args.ckde_scale
    tau_max = args.tau_max

    config = get_config(config_name)
    if kernel_scale is not None:
        config.testing.kernel_scale = [kernel_scale]
    if ckde_scale is not None:
        config.ckde.scale = [ckde_scale]
    if tau_max is not None:
        config.testing.tau_max = [tau_max]
    test(config, test_type, concept_type, workdir)


if __name__ == "__main__":
    args = config()
    main(args)
