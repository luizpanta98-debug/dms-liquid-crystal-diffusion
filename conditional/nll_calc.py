import os
import json
import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from unet import UNet
from dm import DM, stratified_steps
from seed import set_seed
from data import FourCropImageFolder, is_valid_file, get_checkpoint, metrics_dir, apply_epoch_arg


@th.inference_mode()
def calculate_nll(net, dm, loader, config, timesteps=0, exact=10):
	'''Exact bound if timesteps == 0, else the stratified Monte Carlo estimate.'''
	bpd, batch_mean = [], []
	rng = np.random.default_rng(config['seed'])

	for images, labels in tqdm(loader, desc='NLL'):
		images = images.to(config['device']) * 2 - 1  # [0,1] -> [-1,1]
		labels = F.one_hot(
			labels.to(config['device']),
			num_classes=config['label_dim']).float()

		steps = None if timesteps == 0 else stratified_steps(dm.T, timesteps, exact, rng)
		result = dm.nll(images, labels, net, steps=steps)
		bpd.append(result['total_bpd'].cpu().numpy())
		batch_mean.append(float(bpd[-1].mean()))

	return np.concatenate(bpd), np.array(batch_mean)


if __name__ == "__main__":
	with open("config.json", "r") as f:
		config = json.load(f)
	apply_epoch_arg(config)

	# --crops 4 (default): the four corner crops of every held-out image.
	# --crops 1: only the top-left crop, four times faster (one crop per image).
	import argparse
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument('--crops', type=int, choices=[1, 4], default=4)
	parser.add_argument('--batch', type=int, default=config['virtual_batch_size'])
	# --timesteps K: Monte Carlo estimate with the first --exact terms exact and
	# K stratified timesteps for the rest (K + exact network calls per crop
	# instead of T = 1000). 0 (default): exact bound.
	parser.add_argument('--timesteps', type=int, default=0)
	parser.add_argument('--exact', type=int, default=10)
	args, _ = parser.parse_known_args()

	with open(f"{config['save_checkpoint']}/val_files.json", "r") as f:
		val_files = json.load(f)

	set_seed(config['seed'])

	dataset = FourCropImageFolder(
		root=config['data_fp'],
		is_valid_file=is_valid_file)
	dataset.crop_size = config['img_resolution']

	file_to_idx = {
		os.path.abspath(path): i
		for i, (path, _) in enumerate(dataset.samples)
	}
	val_dataset = Subset(dataset, [
		4 * file_to_idx[path] + crop
		for path in val_files for crop in range(args.crops)
	])

	val_loader = DataLoader(
		val_dataset,
		batch_size=args.batch,
		shuffle=False,
		num_workers=min(8, os.cpu_count() or 1),
		pin_memory=True,
	)

	dm = DM(prediction_type=config.get('prediction_type', 'x0'))
	net = UNet(
		img_resolution=config["img_resolution"],
		in_channels=config["in_channels"],
		out_channels=config["out_channels"],
		label_dim=config["label_dim"],
		model_channels=config["model_channels"],
		channel_mult=config["channel_mult"],
		channel_mult_emb=config["channel_mult_emb"],
		num_res_blocks=config["num_res_blocks"],
		attn_resolutions=config["attn_resolutions"],
	).to(config['device'])

	prefix = 'ema_weights' if config['ema'] else 'last_weights'
	model_path, epoch = get_checkpoint(
		f"{config['save_checkpoint']}/weights", prefix, config.get('eval_epoch'))
	net.load_state_dict(th.load(
		model_path, map_location=config['device'], weights_only=False))
	print(f'weights loaded: epoch {epoch}.')
	net.eval()

	set_seed(config['seed'])
	bpd, batch_mean = calculate_nll(net, dm, val_loader, config, args.timesteps, args.exact)
	nll = float(bpd.mean(dtype=np.float64))
	if args.timesteps == 0:
		sem = float(bpd.std(ddof=1) / np.sqrt(len(bpd)))
	else:
		# All crops of a batch share the sampled timesteps, so batches (not
		# crops) are the independent units of the Monte Carlo estimate.
		sem = float(batch_mean.std(ddof=1) / np.sqrt(len(batch_mean)))

	report = {
		'metric': 'conditional variational NLL bound',
		'split': 'validation',
		'nll_bpd': nll,
		'nll_bpd_standard_error': sem,
		'estimator': 'exact' if args.timesteps == 0 else
			f'stratified Monte Carlo: {args.exact} exact terms + {args.timesteps} strata',
		'num_samples': len(bpd),
		'model_path': model_path,
		'epoch': epoch,
		'seed': config['seed'],
		'batch_size': args.batch,
		'crops_per_image': args.crops,
		'class_to_idx': dataset.class_to_idx,
		'prediction_type': dm.prediction_type,
		'T': dm.T,
		'beta_1': dm.beta_1,
		'beta_T': dm.beta_T,
	}

	out = metrics_dir(config, epoch)
	np.savez(f"{out}/nll_per_image.npz",
		bpd=bpd, files=np.repeat(np.array(val_files), args.crops))
	with open(f"{out}/nll.json", "w") as f:
		json.dump(report, f, indent=4)

	print(f'Validation NLL bound: {nll:.4f} +- {sem:.4f} bits/dim ({len(bpd)} crops, {report["estimator"]})')