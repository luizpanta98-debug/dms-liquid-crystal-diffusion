import json
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from torchvision.transforms.functional import to_tensor
from torchvision.datasets import ImageFolder

from ..data import is_valid_file


class FourCropImageFolder(ImageFolder):
    def __len__(self):
        return 4 * super().__len__()

    def __getitem__(self, index):
        image, target = super().__getitem__(index // 4)
        size = self.crop_size
        left = (index % 2) * (image.width - size)
        top = (index % 4 // 2) * (image.height - size)
        return to_tensor(image.crop((left, top, left + size, top + size))), target


def load_config():
    base = Path(__file__).resolve().parent

    with (base / "config.json").open() as f:
        config = json.load(f)

    for key in (
        "data_fp", "save_checkpoint", "diffusion_config",
    ):
        config[key] = str((base / config[key]).resolve())

    with open(config["diffusion_config"]) as f:
        diffusion_config = json.load(f)

    checkpoint_dir = base.parent / diffusion_config["save_checkpoint"]
    config["test_files"] = str((checkpoint_dir / "val_files.json").resolve())
    config["denoiser_weights"] = str((checkpoint_dir / "weights").resolve())

    return config


def path_key(fp):
    return os.path.normcase(os.path.abspath(fp))


def preprocess(images):
    # Shared ImageNet normalization for real and generated images.
    mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (images - mean) / std


def load_data(config, create=False):
    dataset = FourCropImageFolder(
        config["data_fp"],
        is_valid_file=is_valid_file,
    )
    dataset.crop_size = config["img_resolution"]
    dataset.targets = [target for target in dataset.targets for _ in range(4)]

    if len(dataset.classes) != config["label_dim"]:
        raise ValueError("label_dim does not match the dataset classes.")

    lookup = {
        path_key(fp): 4 * index
        for index, (fp, _) in enumerate(dataset.samples)
    }

    with open(config["test_files"]) as f:
        test_files = [path_key(fp) for fp in json.load(f)]

    if not test_files or len(test_files) != len(set(test_files)):
        raise ValueError("The denoiser test list is empty or contains duplicates.")

    if any(fp not in lookup for fp in test_files):
        raise ValueError("A denoiser test file is missing or invalid.")

    split_fp = Path(config["save_checkpoint"]) / "split_files.json"

    if create:
        train_fraction, val_fraction = config["split"]

        if (
            min(train_fraction, val_fraction) <= 0
            or abs(train_fraction + val_fraction - 1.0) > 1e-8
        ):
            raise ValueError("Train and validation fractions must sum to 1.")

        # Reserve the denoiser holdout before splitting the remaining files.
        test_set = set(test_files)
        remaining = [fp for fp in lookup if fp not in test_set]
        labels = [dataset.targets[lookup[fp]] for fp in remaining]

        train_files, val_files = train_test_split(
            remaining,
            test_size=val_fraction,
            stratify=labels,
            random_state=config["seed"],
        )

        split = {
            "train": train_files,
            "val": val_files,
            "test": test_files,
        }

        split_fp.parent.mkdir(parents=True, exist_ok=True)
        with split_fp.open("w") as f:
            json.dump(split, f, indent=4)

    else:
        with split_fp.open() as f:
            split = json.load(f)

        if split["test"] != test_files:
            raise ValueError("The denoiser test list changed since training.")

    subsets = [
        Subset(dataset, [lookup[fp] + crop for fp in split[name] for crop in range(4)])
        for name in ("train", "val", "test")
    ]

    return (*subsets, dataset.classes)
