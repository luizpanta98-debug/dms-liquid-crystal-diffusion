import json
import sys
from pathlib import Path

# Allow direct execution from resnet50/.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = f"{Path(__file__).resolve().parents[1].name}.resnet50"

import numpy as np
import torch as th
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

from .data import load_config, load_data, preprocess
from ..seed import set_seed


def make_net(label_dim, pretrained=True):
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    net = resnet50(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, label_dim)
    return net


def make_loader(dataset, config, sampler=None):
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        sampler=sampler,
        num_workers=config["num_workers"],
        pin_memory=th.device(config["device"]).type == "cuda",
    )


def run_epoch(net, loader, device, opt=None):
    net.train(opt is not None)
    total_loss = 0.0
    correct = 0
    total = 0

    with th.set_grad_enabled(opt is not None):
        for images, labels in tqdm(loader, leave=False):
            images = preprocess(images.to(device))
            labels = labels.to(device)

            if opt is not None:
                opt.zero_grad(set_to_none=True)

            logits = net(images)
            loss = nn.functional.cross_entropy(logits, labels)

            if opt is not None:
                loss.backward()
                opt.step()

            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)

    return total_loss / total, correct / total


def train(net, train_loader, val_loader, config, classes):
    out = Path(config["save_checkpoint"])
    weights = out / "weights"
    weights.mkdir(parents=True, exist_ok=True)

    with (out / "config.json").open("w") as f:
        json.dump(config, f, indent=4)

    opt = th.optim.AdamW(
        net.parameters(),
        lr=config["lr"],
        weight_decay=1e-4,
    )

    best_val = float("inf")
    history = []

    for epoch in range(1, config["epochs"] + 1):
        loss, accuracy = run_epoch(
            net, train_loader, config["device"], opt
        )
        val_loss, val_accuracy = run_epoch(
            net, val_loader, config["device"]
        )

        checkpoint = {
            "weights": net.state_dict(),
            "classes": classes,
            "epoch": epoch,
        }

        th.save(checkpoint, weights / "last_weights.pt")

        if val_loss < best_val:
            best_val = val_loss
            th.save(checkpoint, weights / "best_weights.pt")

        history.append({
            "epoch": epoch,
            "loss": loss,
            "val_loss": val_loss,
            "accuracy": accuracy,
            "val_accuracy": val_accuracy,
        })

        with (out / "history.json").open("w") as f:
            json.dump(history, f, indent=4)

        print(
            f"\nepoch {epoch}"
            f"\nloss {loss:.4f}"
            f"\nval {val_loss:.4f}"
            f"\naccuracy {accuracy:.4f}"
            f"\nval accuracy {val_accuracy:.4f}\n"
        )


if __name__ == "__main__":
    config = load_config()
    set_seed(config["seed"])

    train_dataset, val_dataset, test_dataset, classes = load_data(
        config, create=True
    )

    targets = np.array([
        train_dataset.dataset.targets[i]
        for i in train_dataset.indices
    ])
    counts = np.bincount(targets, minlength=config["label_dim"])

    if np.any(counts == 0):
        raise ValueError("Every class must appear in the training split.")

    sampler = WeightedRandomSampler(
        th.as_tensor(1.0 / counts[targets], dtype=th.double),
        num_samples=len(targets),
        replacement=True,
    )

    net = make_net(config["label_dim"]).to(config["device"])

    print(
        f"train data size: {len(train_dataset)}"
        f"\nval data size: {len(val_dataset)}"
        f"\ntest data size: {len(test_dataset)}"
    )

    train(
        net,
        make_loader(train_dataset, config, sampler),
        make_loader(val_dataset, config),
        config,
        classes,
    )
