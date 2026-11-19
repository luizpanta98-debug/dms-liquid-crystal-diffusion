'''
Memorisation check: are generated textures copies of training crops?

For each generated image we find the closest training crop in Inception feature
space (cosine distance), and compare these distances with the same quantity for
real held-out crops. If generated images were copies, their nearest-neighbour
distances would be much smaller than those of unseen real images.

Training crops are taken on a regular grid (GRID x GRID positions per image),
which approximates the random crops seen during training.

Run from conditional/:  python nearest_neighbours.py
Outputs in <save_checkpoint>/metrics/:
    nearest_neighbours.json   distance statistics per class
    nearest_neighbours.pdf    generated images next to their nearest training crop
    nn_summary.pdf            median distances per phase (plot_nn_summary.py)
'''
import json
import os

import numpy as np
import torch as th
import torch.nn.functional as F
import torch.multiprocessing
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from dm import DM
from seed import set_seed
from data import is_valid_file, metrics_dir
from plot_fid import (
	load_config, build_net, get_inception_model, inception_features)

# Avoid exhausting file descriptors when many batches pass through workers.
torch.multiprocessing.set_sharing_strategy('file_system')

PER_CLASS = 20      # generated images per class
GRID = 3            # training crops per image side (3 x 3 = 9 crops)
SHOW_PER_CLASS = 2  # pairs shown in the figure


class GridCrops(Dataset):
	'''Crops on a GRID x GRID lattice of selected images (uint8 CHW, label, index).'''
	def __init__(self, dataset, indices, size, grid):
		self.dataset, self.indices, self.size, self.grid = dataset, indices, size, grid

	def __len__(self):
		return len(self.indices) * self.grid**2

	def __getitem__(self, i):
		src, pos = divmod(i, self.grid**2)
		image, label = self.dataset[self.indices[src]]
		step = (image.width - self.size) // max(self.grid - 1, 1)
		top, left = divmod(pos, self.grid)
		box = (left*step, top*step, left*step + self.size, top*step + self.size)
		return pil_to_tensor(image.crop(box)), label


@th.inference_mode()
def features(loader, model, device, desc):
	feats, labels = [], []
	for imgs, y in tqdm(loader, desc=desc):
		f = inception_features(imgs.to(device), model)
		feats.append(f)
		# Labels arrive as a plain list (see collate); keeping worker tensors
		# alive would hold one open file descriptor per batch.
		labels.extend(y)
		del imgs
	return np.concatenate(feats), np.asarray(labels)


def collate(batch):
	'''Stack images; keep labels as a plain list (no shared-memory tensor).'''
	return th.stack([b[0] for b in batch]), [int(b[1]) for b in batch]


def normalise(x):
	return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def nearest(query, bank, chunk=1024):
	'''Cosine distance and index of the nearest bank vector for each query.'''
	q, b = th.tensor(normalise(query)), th.tensor(normalise(bank))
	dist, idx = [], []
	for part in q.split(chunk):
		sim = part @ b.T
		s, i = sim.max(1)
		dist.append(1 - s)
		idx.append(i)
	return th.cat(dist).numpy(), th.cat(idx).numpy()


def main():
	print('nearest_neighbours.py version 2026-10-29d')
	config = load_config()
	device = config['device']
	set_seed(config['seed'])
	# Load the model first, so a missing checkpoint fails before the long feature pass.
	dm = DM(prediction_type=config.get('prediction_type', 'x0'),
		guidance_scale=config.get('guidance_scale', 1.5))
	net, model_path, epoch = build_net(config)
	out = metrics_dir(config, epoch)

	dataset = ImageFolder(root=config['data_fp'], is_valid_file=is_valid_file)
	with open(f"{config['save_checkpoint']}/val_files.json") as f:
		held_out = {os.path.abspath(p) for p in json.load(f)}
	train_idx = [i for i, (p, _) in enumerate(dataset.samples) if os.path.abspath(p) not in held_out]
	test_idx = [i for i, (p, _) in enumerate(dataset.samples) if os.path.abspath(p) in held_out]

	model = get_inception_model(device)
	size = config['img_resolution']
	kw = dict(batch_size=50, num_workers=min(8, os.cpu_count() or 1), collate_fn=collate)
	train_set = GridCrops(dataset, train_idx, size, GRID)
	bank, bank_y = features(DataLoader(train_set, **kw), model, device, 'training crops')
	test_feats, test_y = features(DataLoader(GridCrops(dataset, test_idx, size, 2), **kw),
		model, device, 'held-out crops')

	labels = np.repeat(np.arange(config['label_dim']), PER_CLASS)
	set_seed(config['seed'])
	num_steps = config.get('num_steps', 100)
	images = generate_images(net, dm, config, num_steps, labels)
	with th.inference_mode():
		gen_feats = np.concatenate([
			inception_features(batch.to(device), model)
			for batch in images.split(50)])

	d_gen, i_gen = nearest(gen_feats, bank)
	d_test, _ = nearest(test_feats, bank)

	classes = dataset.classes
	report = {
		'checkpoint': model_path, 'epoch': epoch, 'num_steps': num_steps, 'grid': GRID,
		'training_crops': len(train_set), 'held_out_crops': len(test_feats),
		'generated': len(labels), 'distance': 'cosine, Inception pool3 features',
		'generated_median': float(np.median(d_gen)),
		'held_out_median': float(np.median(d_test)),
		'fraction_generated_below_held_out_1st_percentile':
			float(np.mean(d_gen < np.percentile(d_test, 1))),
		'per_class': {
			classes[c]: {
				'generated_median': float(np.median(d_gen[labels == c])),
				'held_out_median': float(np.median(d_test[test_y == c])),
			} for c in range(config['label_dim'])},
	}
	with open(os.path.join(out, 'nearest_neighbours.json'), 'w') as f:
		json.dump(report, f, indent=4)
	print(json.dumps(report, indent=4))

	# The numbers are already saved; a failing figure must not stop the run.
	def plot_summary_figure():
		from plot_nn_summary import main as plot_summary
		plot_summary(os.path.join(out, 'nearest_neighbours.json'))
	for name, job in (
			('nearest_neighbours.pdf', lambda: plot_pairs(images, labels, i_gen, d_gen, train_set,
				classes, os.path.join(out, 'nearest_neighbours.pdf'))),
			('nn_summary.pdf', plot_summary_figure)):
		try:
			job()
		except Exception as e:
			print(f'WARNING: could not draw {name}: {type(e).__name__}: {e}')


@th.inference_mode()
def generate_images(net, dm, config, num_steps, labels):
	dev = config['device']
	subseq = np.linspace(dm.T-1, 0, num_steps, dtype=int).tolist()
	gen = th.Generator(device=dev).manual_seed(config['seed'])
	images = []
	for start in range(0, len(labels), 10):
		y = th.tensor(labels[start:start+10], dtype=th.long, device=dev)
		y = F.one_hot(y, num_classes=config['label_dim']).float()
		xt = th.randn(len(y), config['in_channels'], config['img_resolution'],
			config['img_resolution'], device=dev, generator=gen)
		for k, time in enumerate(subseq):
			ts = th.full((len(y),), time, dtype=th.long, device=dev)
			if k < len(subseq)-1:
				prev = th.full((len(y),), subseq[k+1], dtype=th.long, device=dev)
				xt, _, _ = dm.ddim_sample(config.get('eta', 1.0), net, xt, ts, prev, y)
			else:
				xt, _ = dm.ddim_pred_x0(net, xt, ts, y)
				xt = xt.clamp(-1, 1)
		images.append(((xt+1)*127.5).clamp(0, 255).to(th.uint8).cpu())
	return th.cat(images)


def plot_pairs(images, labels, idx, dist, train_set, classes, path):
	"""Grid: one column per phase; rows alternate generated / nearest training crop."""
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	n = len(classes)
	fig, axes = plt.subplots(2*SHOW_PER_CLASS, n, figsize=(1.0*n, 2.1*SHOW_PER_CLASS))
	for c in range(n):
		chosen = np.flatnonzero(labels == c)[:SHOW_PER_CLASS]
		for k, i in enumerate(chosen):
			real, _ = train_set[int(idx[i])]
			top, bottom = axes[2*k, c], axes[2*k+1, c]
			top.imshow(images[i].permute(1, 2, 0).numpy())
			bottom.imshow(real.permute(1, 2, 0).numpy())
			bottom.set_xlabel(f'd = {dist[i]:.3f}', fontsize=6, labelpad=1)
		axes[0, c].set_title(classes[c], fontsize=7)
	for k in range(SHOW_PER_CLASS):
		axes[2*k, 0].set_ylabel('Generated', fontsize=6)
		axes[2*k+1, 0].set_ylabel('Nearest\ntraining', fontsize=6)
	for ax in axes.ravel():
		ax.set_xticks([]); ax.set_yticks([])
	fig.tight_layout(pad=0.3)
	fig.savefig(path, dpi=200)
	plt.close(fig)
