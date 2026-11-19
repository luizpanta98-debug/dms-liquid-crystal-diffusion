import json
import sys
from pathlib import Path

# Allow direct execution from resnet50/.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = f"{Path(__file__).resolve().parents[1].name}.resnet50"

import numpy as np
import torch as th
import torch.nn.functional as F
from torchvision.datasets.folder import find_classes
from tqdm import tqdm

from ..dm import DM
from ..unet import UNet
from .data import load_config, load_data, preprocess
from ..seed import set_seed
from ..data import get_checkpoint
from .train_script import make_net, make_loader
from .conf_matrix import evaluate, save_comparison


def load_denoiser(config, classes):
    with open(config["diffusion_config"]) as f:
        diffusion_config = json.load(f)

    root = Path(__file__).resolve().parents[1]
    data_fp = root / diffusion_config["data_fp"]
    diffusion_classes, _ = find_classes(str(data_fp))

    if diffusion_classes != classes:
        raise ValueError("Classifier and denoiser class mappings differ.")

    if (
        diffusion_config["label_dim"] != config["label_dim"]
        or diffusion_config["img_resolution"] != config["img_resolution"]
        or diffusion_config["in_channels"] != 3
        or diffusion_config["out_channels"] != 3
    ):
        raise ValueError("Classifier and denoiser image or label settings differ.")

    keys = (
        "img_resolution", "in_channels", "out_channels",
        "label_dim", "model_channels", "channel_mult",
        "channel_mult_emb", "num_res_blocks", "attn_resolutions",
    )

    net = UNet(**{
        key: diffusion_config[key] for key in keys
    }).to(config["device"])

    prefix = "ema_weights" if diffusion_config["ema"] else "last_weights"
    # resnet50/config.json "denoiser_epoch" overrides "eval_epoch" of the
    # diffusion config; null in both loads the latest checkpoint.
    epoch = config.get("denoiser_epoch")
    if epoch is None:
        epoch = diffusion_config.get("eval_epoch")
    # Sampling settings (num_steps, eta, guidance) come from the diffusion config.
    config["num_steps"] = diffusion_config.get("num_steps", 100)
    config["eta"] = diffusion_config.get("eta", 1.0)
    model_path, epoch = get_checkpoint(config["denoiser_weights"], prefix, epoch)
    net.load_state_dict(th.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    ))
    net.eval()
    print(f"Denoiser weights loaded: epoch {epoch}.")

    dm = DM(
        prediction_type=diffusion_config.get("prediction_type", "x0"),
        guidance_scale=diffusion_config.get("guidance_scale", 1.5),
    )
    metadata = {
        "checkpoint": str(Path(model_path).resolve()),
        "epoch": epoch,
        "config": diffusion_config,
        "prediction_type": dm.prediction_type,
        "guidance_scale": dm.guidance_scale,
        "sampler": "DDIM",
        "num_steps": config["num_steps"],
        "eta": config["eta"],
    }
    return net, dm, metadata


@th.inference_mode()
def sample(net, dm, labels, config):
    if not 1 <= config["num_steps"] <= dm.T:
        raise ValueError("num_steps must be between 1 and dm.T.")

    if not 0 <= config["eta"] <= 1:
        raise ValueError("eta must be between 0 and 1.")

    labels = labels.to(config["device"])
    y = F.one_hot(labels, num_classes=config["label_dim"]).float()

    xt = th.randn(
        len(labels), 3,
        config["img_resolution"], config["img_resolution"],
        device=config["device"],
    )

    times = np.linspace(
        dm.T - 1, 0, config["num_steps"], dtype=int
    ).tolist()

    for i, time in enumerate(times):
        ts = th.full_like(labels, time)

        if i + 1 < len(times):
            ts_prev = th.full_like(labels, times[i + 1])
            xt, _, _ = dm.ddim_sample(
                config["eta"], net, xt, ts, ts_prev, y
            )
        else:
            xt, _ = dm.ddim_pred_x0(net, xt, ts, y)

    images = (xt.clamp(-1, 1) + 1) / 2

    # Match the uint8 conversion used by the existing PNG sampler.
    return (images * 255).floor() / 255


@th.inference_mode()
def predict_real(net, loader, device):
    targets = []
    probabilities = []

    for images, labels in tqdm(loader, desc="Real samples"):
        images = preprocess(images.to(device))
        probabilities.append(net(images).softmax(1).cpu())
        targets.append(labels)

    return (
        th.cat(targets).numpy(),
        th.cat(probabilities).numpy(),
    )


@th.inference_mode()
def predict_synthetic(classifier, denoiser, dm, targets, config):
    probabilities = []
    batches = th.tensor(targets, dtype=th.long).split(
        config["sample_batch_size"]
    )

    with tqdm(total=len(targets), desc="Synthetic samples", unit="img") as progress:
        for labels in batches:
            images = sample(denoiser, dm, labels, config)
            probabilities.append(
                classifier(preprocess(images)).softmax(1).cpu()
            )
            progress.update(len(labels))

    return th.cat(probabilities).numpy()


if __name__ == "__main__":
    config = load_config()
    set_seed(config["seed"])

    _, _, test_dataset, classes = load_data(config)

    checkpoint = th.load(
        Path(config["save_checkpoint"]) / "weights/best_weights.pt",
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["classes"] != classes:
        raise ValueError("Checkpoint class mapping differs from the dataset.")

    classifier_epoch = checkpoint["epoch"]

    classifier = make_net(
        config["label_dim"], pretrained=False
    ).to(config["device"])
    classifier.load_state_dict(checkpoint["weights"])
    classifier.eval()
    del checkpoint

    targets, real_prob = predict_real(
        classifier,
        make_loader(test_dataset, config),
        config["device"],
    )

    denoiser, dm, denoiser_metadata = load_denoiser(config, classes)
    set_seed(config["seed"])

    # One generated image for each real-test label.
    gen_prob = predict_synthetic(
        classifier, denoiser, dm, targets, config
    )

    # One folder per denoiser run and epoch (x0/eps models and epochs never overwrite each other).
    run_name = Path(denoiser_metadata["config"]["save_checkpoint"]).name
    out = Path(config["save_checkpoint"]) / "test" / run_name / f"epoch_{denoiser_metadata['epoch']}"

    real_metrics, gen_metrics = save_comparison(
        evaluate(targets, real_prob, config["label_dim"]),
        evaluate(targets, gen_prob, config["label_dim"]),
        classes,
        out,
        metadata={
            "classifier_epoch": classifier_epoch,
            "denoiser": denoiser_metadata,
            "config": config,
        },
    )

    np.savez_compressed(
        out / "predictions.npz",
        targets=targets,
        real_probabilities=real_prob,
        synthetic_probabilities=gen_prob,
    )

    for name, metrics in (
        ("real", real_metrics),
        ("synthetic", gen_metrics),
    ):
        print(f"\n{name}")
        for metric, value in metrics.items():
            text = f"{value:.4f}" if value is not None else "N/A"
            print(f"{metric}: {text}")

    print(f"\nresults saved -> {out}")
