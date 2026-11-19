'''
Ordinal-pattern comparison of generated and real textures.

Physics-style, interpretable descriptors computed on the grayscale image
(ordinal.py):
  * the 75 tie-aware 2x2 ordinal patterns and their 11 groups A-K
    (Voltarelli et al., arXiv:2603.26723), classified with a Hellinger 1-NN;
  * the permutation entropy H and statistical complexity C of the strict
    2x2 Bandt-Pompe patterns (Sigaki et al., PRE 99, 013311, 2019).

What is compared:
  real training bank   four 256x256 corner crops of up to --train_per_class
                       training images per class (random, seed config['seed'])
  real held-out set    four corner crops of every held-out image
                       (<save_checkpoint>/val_files.json, as nll_calc.py)
  generated set        --per_class DDIM images per class at the chosen epoch
                       (num_steps, eta, guidance_scale from config.json),
                       quantised to uint8 like the other scripts

Analyses:
  (i)   1-NN phase classifier (bank = real training crops), applied to the
        held-out crops and to the generated images (true label = requested
        phase); 75 patterns and 11 groups.
  (ii)  Hellinger distance between class-mean pattern distributions:
        generated phase i vs real held-out phase j, and the baseline
        real held-out vs real training.
  (iii) Complexity-entropy plane: per-phase mean and std of H and C.

Usage (from conditional/, on the GPU machine):
	python ordinal_analysis.py --epoch 100
	python ordinal_analysis.py --epoch 100 --generated_dir <metrics>/ordinal/generated
Options: --per_class 200  --train_per_class 300  --batch 16  --precision fp32|bf16
	--workers 8
Generated PNGs are written to <metrics>/ordinal/generated/<Class>/xxxxx.png;
existing files are kept, so an interrupted run can simply be restarted.

Outputs in <save_checkpoint>/metrics/epoch_<E>/ordinal/:
	ordinal_metrics.json      all numbers, settings, checkpoint and counts
	features.npz              per-image features (for further analysis)
	ordinal_confusion.pdf     75-pattern 1-NN, (a) real held-out, (b) generated
	ordinal_confusion_groups.pdf   the same for the 11 groups
	hc_plane.pdf              complexity-entropy plane, one panel per phase
	ordinal_distance.pdf      generated vs real Hellinger distance matrix
(every figure also as .png)

Everything except model loading and generation is importable without torch
(see test_ordinal.py).
'''
import argparse
import json
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from ordinal import (GROUP_NAMES, to_gray, two_by_two_distribution, group_distribution,
	permutation_distribution, entropy_complexity, hellinger, nearest_neighbour)

DX, DY = 2, 2          # Bandt-Pompe window for H and C (Sigaki 2019, simulated nematics)
GEN_BATCH = 16         # generated images per batch (guidance doubles it inside dm.py)
ACCENT = '#2a78d6'     # generated images in all figures
REAL_GRAY = '#b4b4b4'  # real held-out images


# ---------------------------------------------------------------- features

def image_features(rgb):
	'''uint8 RGB image -> (75 patterns, 11 groups, (H, C)).'''
	gray = to_gray(rgb)
	p75 = two_by_two_distribution(gray)
	H, C = entropy_complexity(permutation_distribution(gray, dx=DX, dy=DY))
	return p75, group_distribution(p75), np.array([H, C])


def corner_crops(image, size):
	'''The four size x size corner crops of a PIL image, as uint8 arrays.

	Same order as data.FourCropImageFolder: top-left, top-right, bottom-left,
	bottom-right.
	'''
	crops = []
	for k in range(4):
		left = (k % 2) * (image.width - size)
		top = (k // 2) * (image.height - size)
		crops.append(np.asarray(image.crop((left, top, left + size, top + size))))
	return crops


def file_features(job):
	'''Features of one file: its four corner crops (size given) or the whole image (size None).'''
	path, size = job
	with Image.open(path) as f:
		image = f.convert('RGB')
	crops = [np.asarray(image)] if size is None else corner_crops(image, size)
	return [image_features(c) for c in crops]


def feature_table(paths, labels, size, workers=1, desc='features'):
	'''Features of many files (in parallel). Returns dict of arrays p75, p11, hc, y.'''
	jobs = [(p, size) for p in paths]
	if workers > 1:
		with Pool(workers) as pool:
			results = list(_progress(pool.imap(file_features, jobs, chunksize=8), len(jobs), desc))
	else:
		results = [file_features(j) for j in _progress(jobs, len(jobs), desc)]
	rows = [(f, y) for res, y in zip(results, labels) for f in res]
	if not rows:
		return {'p75': np.zeros((0, 75)), 'p11': np.zeros((0, 11)),
			'hc': np.zeros((0, 2)), 'y': np.zeros(0, int)}
	return {
		'p75': np.stack([f[0] for f, _ in rows]),
		'p11': np.stack([f[1] for f, _ in rows]),
		'hc': np.stack([f[2] for f, _ in rows]),
		'y': np.array([y for _, y in rows]),
	}


def _progress(iterable, total, desc):
	try:
		from tqdm import tqdm
		return tqdm(iterable, total=total, desc=desc)
	except ImportError:
		return iterable


# ---------------------------------------------------------------- analyses

def normalized_hundredths(counts):
	'''Row proportions rounded to hundredths, each row summing to exactly 1.

	Copy of resnet50/conf_matrix.normalized_hundredths (that module needs
	sklearn): leftover hundredths go to the largest remainders. Empty rows: NaN.
	'''
	rounded = np.full(counts.shape, np.nan)
	for i, row in enumerate(counts):
		total = int(row.sum())
		if total == 0:
			continue
		quotients = [divmod(int(c) * 100, total) for c in row]
		hundredths = [q for q, _ in quotients]
		order = sorted(range(len(row)), key=lambda j: (-quotients[j][1], j))
		for j in order[:100 - sum(hundredths)]:
			hundredths[j] += 1
		rounded[i] = np.array(hundredths) / 100
	return rounded


def classification_metrics(y_true, y_pred, n):
	'''Confusion matrix and summary scores (same definitions as resnet50/conf_matrix.py).'''
	counts = np.zeros((n, n), dtype=int)
	np.add.at(counts, (y_true, y_pred), 1)
	support = counts.sum(axis=1)
	with np.errstate(invalid='ignore', divide='ignore'):
		matrix = counts / support[:, None]          # NaN rows for absent classes
	tp = np.diag(counts)
	denom = 2 * tp + (counts.sum(axis=0) - tp) + (support - tp)
	f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
	complete = bool(np.all(support > 0))
	return counts, matrix, {
		'accuracy': float(tp.sum() / max(counts.sum(), 1)),
		'balanced_accuracy': float(np.diag(matrix).mean()) if complete else None,
		'macro_f1': float(f1.mean()) if complete else None,
		'recall_per_class': [None if s == 0 else float(r) for s, r in zip(support, np.diag(matrix))],
	}


def class_means(P, y, n):
	'''Mean distribution of each class (NaN row if a class has no images).'''
	return np.array([P[y == c].mean(axis=0) if np.any(y == c) else np.full(P.shape[1], np.nan)
		for c in range(n)])


def distance_matrix(P_rows, y_rows, P_cols, y_cols, n):
	'''D[i, j] = Hellinger(mean of row class i, mean of column class j).'''
	A, B = class_means(P_rows, y_rows, n), class_means(P_cols, y_cols, n)
	D = hellinger(A[:, None, :], B[None, :, :])
	diag_is_min = [bool(np.isfinite(D[i]).all() and np.argmin(D[i]) == i) for i in range(n)]
	return D, diag_is_min


def hc_summary(hc, y, classes):
	out = {}
	for c, name in enumerate(classes):
		v = hc[y == c]
		out[name] = {'n': int(len(v)),
			'H_mean': float(v[:, 0].mean()) if len(v) else None,
			'H_std': float(v[:, 0].std()) if len(v) else None,
			'C_mean': float(v[:, 1].mean()) if len(v) else None,
			'C_std': float(v[:, 1].std()) if len(v) else None}
	return out


def analyse(bank, real, gen, classes):
	'''All three analyses. bank/real/gen: dicts from feature_table.

	Returns (report dict for the JSON file, arrays needed by the figures).
	'''
	n = len(classes)
	report, arrays = {}, {}
	for key, name in (('p75', 'patterns75'), ('p11', 'groups11')):
		part = {}
		for set_name, data in (('real', real), ('synthetic', gen)):
			pred, dist = nearest_neighbour(bank[key], bank['y'], data[key])
			counts, matrix, metrics = classification_metrics(data['y'], pred, n)
			metrics['nn_distance_median'] = float(np.median(dist)) if len(dist) else None
			metrics['confusion_counts'] = counts.tolist()
			metrics['confusion_normalized'] = _nan_to_none(matrix)
			part[set_name] = metrics
			arrays[f'{key}_{set_name}_counts'] = counts
			arrays[f'{key}_{set_name}_matrix'] = matrix
		# Mean total-variation distance between corresponding rows (as for ResNet-50).
		diff = np.abs(arrays[f'{key}_real_matrix'] - arrays[f'{key}_synthetic_matrix'])
		part['confusion_distance'] = float(diff.sum() / (2 * n)) if np.isfinite(diff).all() else None
		report[name] = part

	distances = {}
	for key, name in (('p75', 'patterns75'), ('p11', 'groups11')):
		D_gen, ok_gen = distance_matrix(gen[key], gen['y'], real[key], real['y'], n)
		D_base, ok_base = distance_matrix(real[key], real['y'], bank[key], bank['y'], n)
		distances[name] = {
			'generated_vs_real_heldout': _nan_to_none(D_gen),
			'generated_diagonal_is_row_min': ok_gen,
			'generated_diagonal_min_count': int(sum(ok_gen)),
			'baseline_heldout_vs_train': _nan_to_none(D_base),
			'baseline_diagonal_is_row_min': ok_base,
			'baseline_diagonal_min_count': int(sum(ok_base)),
		}
		arrays[f'{key}_distance'] = D_gen
	report['distance_matrices'] = distances
	report['hc_plane'] = {'real_heldout': hc_summary(real['hc'], real['y'], classes),
		'synthetic': hc_summary(gen['hc'], gen['y'], classes)}
	return report, arrays


def _nan_to_none(matrix):
	return [[None if not np.isfinite(v) else float(v) for v in row] for row in np.asarray(matrix)]


# ---------------------------------------------------------------- figures

def serif_style():
	'''Font settings of the article figures (as in collect_results.plot).'''
	from matplotlib import font_manager
	for f in font_manager.findSystemFonts():
		if 'texgyretermes' in f.lower():
			font_manager.fontManager.addfont(f)
	return {'font.family': 'serif',
		'font.serif': ['TeX Gyre Termes', 'Times New Roman', 'DejaVu Serif'],
		'font.size': 8, 'mathtext.fontset': 'stix', 'pdf.fonttype': 42}


def save(fig, path):
	fig.savefig(path, bbox_inches='tight', pad_inches=0.04)
	fig.savefig(str(path).rsplit('.', 1)[0] + '.png', dpi=300, bbox_inches='tight', pad_inches=0.04)


def plot_confusion(arrays, key, classes, folder, out):
	'''Vertical (a) real / (b) synthetic confusion figure in the article style.

	article_plots.plot_confusion reads <folder>/confusion_matrices.npz (the format written
	by resnet50/conf_matrix.py), so we write that file and reuse the same code.
	'''
	folder = Path(folder)
	folder.mkdir(parents=True, exist_ok=True)
	real, gen = arrays[f'{key}_real_counts'], arrays[f'{key}_synthetic_counts']
	np.savez_compressed(folder / 'confusion_matrices.npz', classes=np.asarray(classes),
		real_counts=real, synthetic_counts=gen,
		real_normalized=arrays[f'{key}_real_matrix'],
		synthetic_normalized=arrays[f'{key}_synthetic_matrix'],
		real_display_normalized=normalized_hundredths(real),
		synthetic_display_normalized=normalized_hundredths(gen))
	from article_plots import plot_confusion as plot_article_confusion
	plot_article_confusion(folder, str(out))


def complexity_bounds(n, points=400):
	'''Minimum and maximum C as a function of H for n patterns (Martin et al. 2006).

	C_min: one probability p, the others (1-p)/(n-1), p in [1/n, 1].
	C_max: m patterns absent, one probability p, the others (1-p)/(n-m-1),
	with p in [0, 1/(n-m)], for m = 0 .. n-2.
	'''
	lower = []
	for p in np.linspace(1/n, 1, points):
		lower.append(entropy_complexity(np.r_[p, np.full(n - 1, (1 - p) / (n - 1))]))
	upper = []
	for m in range(n - 1):
		for p in np.linspace(0, 1/(n - m), points):
			rest = np.full(n - m - 1, (1 - p) / (n - m - 1))
			upper.append(entropy_complexity(np.r_[np.zeros(m), p, rest]))
	upper = np.array(upper)
	# Upper envelope: largest C in each small bin of H.
	bins = np.minimum((upper[:, 0] * points).astype(int), points - 1)
	envelope = [upper[bins == b][np.argmax(upper[bins == b, 1])] for b in np.unique(bins)]
	return np.array(lower), np.array(envelope)


def plot_hc_plane(real, gen, classes, out):
	'''Small multiples: one panel per phase, H on x, C on y, shared axes.'''
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	n = len(classes)
	cols = 4
	rows = int(np.ceil((n + 1) / cols))         # one free cell for the legend
	both = np.concatenate([real['hc'], gen['hc']])
	lo, hi = both.min(axis=0), both.max(axis=0)
	pad = 0.06 * np.maximum(hi - lo, 1e-3)
	lower, upper = complexity_bounds(math.factorial(DX * DY))
	with plt.rc_context(serif_style()):
		fig, axes = plt.subplots(rows, cols, figsize=(7.0, 1.75 * rows + 0.2),
			sharex=True, sharey=True)
		axes = axes.ravel()
		for c, ax in enumerate(axes[:n]):
			for curve in (lower, upper):
				ax.plot(curve[:, 0], curve[:, 1], color='#d8d6d0', lw=0.6, zorder=0)
			r, g = real['hc'][real['y'] == c], gen['hc'][gen['y'] == c]
			ax.scatter(r[:, 0], r[:, 1], s=3, color=REAL_GRAY, alpha=0.6, lw=0, label='Real')
			ax.scatter(g[:, 0], g[:, 1], s=3, color=ACCENT, alpha=0.5, lw=0, label='Generated')
			for v, color, label in ((r, '#555555', 'Real mean'), (g, ACCENT, 'Generated mean')):
				if len(v):
					ax.scatter(*v.mean(axis=0), s=36, color=color, edgecolor='white', lw=0.8,
						zorder=3, marker='o' if color == ACCENT else 's', label=label)
			ax.set_title(classes[c], fontsize=8, pad=2)
			ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
			ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
			ax.grid(True, color='#eeede9', lw=0.5)
			ax.set_axisbelow(True)
			ax.tick_params(length=2, labelsize=7)
			for side in ('top', 'right'):
				ax.spines[side].set_visible(False)
		for k, ax in enumerate(axes):
			if k % cols == 0:
				ax.set_ylabel('Complexity $C$')
			if k < n:
				# x tick labels on every panel (shared x axis hides them by default)
				ax.xaxis.set_tick_params(labelbottom=True)
			if k >= n - cols:
				ax.set_xlabel('Entropy $H$')
		handles, labels = axes[0].get_legend_handles_labels()
		for ax in axes[n:]:
			ax.axis('off')
		if n < len(axes):
			axes[n].legend(handles, labels, loc='center', frameon=False, fontsize=7,
				markerscale=1.0, handletextpad=0.3)
		fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.6)
		save(fig, out)
		plt.close(fig)


def plot_distance(D, classes, out, title=None):
	'''7x7 heatmap of Hellinger distances; row minimum outlined.'''
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	from matplotlib.patches import Rectangle
	n = len(classes)
	finite = D[np.isfinite(D)]
	vmax = finite.max() if len(finite) else 1.0
	with plt.rc_context(serif_style()):
		fig, ax = plt.subplots(figsize=(3.6, 3.2))
		cmap = plt.get_cmap('Blues_r').with_extremes(bad='#EEEEEE')
		image = ax.imshow(np.ma.masked_invalid(D), cmap=cmap, vmin=0, vmax=vmax)
		ax.set(xticks=np.arange(n), yticks=np.arange(n),
			xlabel='Real held-out phase', ylabel='Generated (requested) phase')
		ax.set_xticklabels(classes, rotation=45, ha='right', rotation_mode='anchor')
		ax.set_yticklabels(classes)
		if title:
			ax.set_title(title)
		ax.tick_params(length=0)
		for spine in ax.spines.values():
			spine.set_visible(False)
		for i in range(n):
			row = D[i]
			for j in range(n):
				v = row[j]
				ax.text(j, i, f'{v:.3f}' if np.isfinite(v) else '—', ha='center', va='center',
					fontsize=6.5, color='white' if np.isfinite(v) and v < 0.45 * vmax else 'black')
			if np.isfinite(row).all():
				j = int(np.argmin(row))
				ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec='#e8a33d', lw=1.4))
		fig.colorbar(image, ax=ax, shrink=0.8, aspect=25, pad=0.03,
			label='Hellinger distance (lower = more similar)')
		fig.tight_layout(pad=0.3)
		save(fig, out)
		plt.close(fig)


def make_figures(report, arrays, real, gen, classes, out):
	out = Path(out)
	jobs = (
		('ordinal_confusion.pdf', lambda: plot_confusion(arrays, 'p75', classes,
			out / 'confusion_patterns75', out / 'ordinal_confusion.pdf')),
		('ordinal_confusion_groups.pdf', lambda: plot_confusion(arrays, 'p11', classes,
			out / 'confusion_groups11', out / 'ordinal_confusion_groups.pdf')),
		('hc_plane.pdf', lambda: plot_hc_plane(real, gen, classes, out / 'hc_plane.pdf')),
		('ordinal_distance.pdf', lambda: plot_distance(arrays['p75_distance'], classes,
			out / 'ordinal_distance.pdf')),
	)
	# The numbers are already saved; a failing figure must not stop the run.
	for name, job in jobs:
		try:
			job()
		except Exception as e:
			print(f'WARNING: could not draw {name}: {type(e).__name__}: {e}')


def write_outputs(report, arrays, bank, real, gen, classes, out):
	out = Path(out)
	out.mkdir(parents=True, exist_ok=True)
	with open(out / 'ordinal_metrics.json', 'w') as f:
		json.dump(report, f, indent=4)
	np.savez_compressed(out / 'features.npz', classes=np.asarray(classes),
		**{f'{s}_{k}': d[k] for s, d in (('bank', bank), ('real', real), ('gen', gen)) for k in d})
	make_figures(report, arrays, real, gen, classes, out)


# ---------------------------------------------------------------- data

def split_files(dataset, val_json):
	'''(train indices, held-out indices) of an ImageFolder, from val_files.json.'''
	with open(val_json) as f:
		held_out = {os.path.abspath(p) for p in json.load(f)}
	train, test = [], []
	for i, (p, _) in enumerate(dataset.samples):
		(test if os.path.abspath(p) in held_out else train).append(i)
	return train, test


def pick_per_class(targets, indices, per_class, seed):
	'''Random subset of at most per_class indices of each class (fixed seed).'''
	rng = np.random.RandomState(seed)
	targets = np.asarray(targets)
	indices = np.asarray(indices)
	chosen = []
	for c in np.unique(targets):
		pool = indices[targets[indices] == c]
		chosen.extend(sorted(rng.choice(pool, size=min(per_class, len(pool)), replace=False)))
	return [int(i) for i in chosen]


def generated_files(folder, classes):
	'''PNG paths and labels in <folder>/<Class>/*.png.'''
	paths, labels = [], []
	for c, name in enumerate(classes):
		files = sorted((Path(folder) / name).glob('*.png'))
		if not files:
			print(f'WARNING: no generated images for {name} in {folder}')
		paths += [str(p) for p in files]
		labels += [c] * len(files)
	return paths, labels


# ---------------------------------------------------------------- generation (torch)

def generate(config, classes, per_class, folder, batch=GEN_BATCH, precision='fp32'):
	'''DDIM samples of every phase, saved as PNG; existing files are skipped.

	Same sampler as nearest_neighbours.generate_images: config['num_steps']
	DDIM steps, eta = config['eta'], guidance from config['guidance_scale'],
	final x0 clamped to [-1, 1] and quantised with (x+1)*127.5 -> uint8.
	Returns (checkpoint path, epoch).
	'''
	import torch as th
	import torch.nn.functional as F
	from dm import DM
	from seed import set_seed
	from plot_fid import build_net

	set_seed(config['seed'])
	dm = DM(prediction_type=config.get('prediction_type', 'x0'),
		guidance_scale=config.get('guidance_scale', 1.5))
	net, model_path, epoch = build_net(config)
	dev = config['device']
	use_amp = precision == 'bf16' and th.device(dev).type == 'cuda'
	subseq = np.linspace(dm.T - 1, 0, config.get('num_steps', 100), dtype=int).tolist()
	eta = config.get('eta', 1.0)
	size = config['img_resolution']

	for c, name in enumerate(classes):
		os.makedirs(Path(folder) / name, exist_ok=True)
		for start in _progress(range(0, per_class, batch), (per_class + batch - 1) // batch, name):
			names = [Path(folder) / name / f'{k:05d}.png' for k in range(start, min(start + batch, per_class))]
			if all(p.exists() for p in names):
				continue
			B = len(names)
			# Initial noise depends only on (seed, class, position): restarts are reproducible.
			gen = th.Generator(device=dev).manual_seed(config['seed'] + 100000 * c + start)
			y = F.one_hot(th.full((B,), c, dtype=th.long, device=dev),
				num_classes=config['label_dim']).float()
			xt = th.randn(B, config['in_channels'], size, size, device=dev, generator=gen)
			with th.inference_mode(), th.autocast('cuda', dtype=th.bfloat16, enabled=use_amp):
				for k, time in enumerate(subseq):
					ts = th.full((B,), time, dtype=th.long, device=dev)
					if k < len(subseq) - 1:
						prev = th.full((B,), subseq[k + 1], dtype=th.long, device=dev)
						xt, _, _ = dm.ddim_sample(eta, net, xt, ts, prev, y)
					else:
						xt, _ = dm.ddim_pred_x0(net, xt, ts, y)
						xt = xt.clamp(-1, 1)
			images = ((xt.float() + 1) * 127.5).clamp(0, 255).to(th.uint8).cpu()
			for img, path in zip(images, names):
				Image.fromarray(img.permute(1, 2, 0).numpy()).save(path)
	return model_path, epoch


# ---------------------------------------------------------------- main

def parse_args(argv=None):
	p = argparse.ArgumentParser(description=__doc__.split('\n\n')[1],
		formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument('--epoch', type=int, default=None, help='checkpoint epoch (else eval_epoch)')
	p.add_argument('--per_class', type=int, default=200, help='generated images per class')
	p.add_argument('--train_per_class', type=int, default=300, help='training images per class in the 1-NN bank')
	p.add_argument('--generated_dir', default=None, help='use these PNGs instead of generating')
	p.add_argument('--batch', type=int, default=GEN_BATCH)
	p.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
	p.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
	return p.parse_args(argv)


def main():
	print('ordinal_analysis.py version 2026-11-11')
	args = parse_args()
	from torchvision.datasets import ImageFolder
	from data import is_valid_file, get_checkpoint, metrics_dir, apply_epoch_arg

	with open('config.json') as f:
		config = apply_epoch_arg(json.load(f))
	size = config['img_resolution']
	dataset = ImageFolder(root=config['data_fp'], is_valid_file=is_valid_file)
	classes = dataset.classes
	train_idx, test_idx = split_files(dataset, f"{config['save_checkpoint']}/val_files.json")
	bank_idx = pick_per_class(dataset.targets, train_idx, args.train_per_class, config['seed'])

	if args.generated_dir is None:
		prefix = 'ema_weights' if config['ema'] else 'last_weights'
		_, epoch = get_checkpoint(f"{config['save_checkpoint']}/weights", prefix, config.get('eval_epoch'))
		gen_dir = os.path.join(metrics_dir(config, epoch), 'ordinal', 'generated')
		model_path, epoch = generate(config, classes, args.per_class, gen_dir,
			args.batch, args.precision)
	else:
		gen_dir = args.generated_dir
		prefix = 'ema_weights' if config['ema'] else 'last_weights'
		model_path, epoch = get_checkpoint(f"{config['save_checkpoint']}/weights", prefix,
			config.get('eval_epoch'))
	out = os.path.join(metrics_dir(config, epoch), 'ordinal')
	os.makedirs(out, exist_ok=True)

	def files(indices):
		return [dataset.samples[i][0] for i in indices], [dataset.samples[i][1] for i in indices]

	bank = feature_table(*files(bank_idx), size, args.workers, 'training crops')
	real = feature_table(*files(test_idx), size, args.workers, 'held-out crops')
	gen_paths, gen_labels = generated_files(gen_dir, classes)
	gen = feature_table(gen_paths, gen_labels, None, args.workers, 'generated images')

	report, arrays = analyse(bank, real, gen, classes)
	report = {
		'checkpoint': model_path, 'epoch': epoch, 'classes': classes,
		'settings': {
			'grayscale': 'Y = 0.2125 R + 0.7154 G + 0.0721 B, float64, from uint8 RGB, not rounded',
			'patterns': '75 tie-aware 2x2 ordinal patterns (dense ranks), unit step, '
				'row-major; order of ordpy.two_by_two_patterns(tie_patterns=True)',
			'groups': GROUP_NAMES,
			'classifier': '1-NN, Hellinger distance, bank = real training crops',
			'hc': f'strict Bandt-Pompe dx={DX}, dy={DY}, ties by occurrence order (stable argsort), '
				'normalised permutation entropy and Jensen-Shannon complexity',
			'crop': f'{size}x{size} corner crops (4 per real image)',
			'num_steps': config.get('num_steps', 100), 'eta': config.get('eta', 1.0),
			'guidance_scale': config.get('guidance_scale', 1.5),
			'prediction_type': config.get('prediction_type'),
			'seed': config['seed'], 'per_class': args.per_class,
			'train_per_class': args.train_per_class, 'generated_dir': os.path.abspath(gen_dir),
			'precision': args.precision if args.generated_dir is None else None,
		},
		'counts': {
			'bank_images': len(bank_idx), 'bank_crops': int(len(bank['y'])),
			'heldout_images': len(test_idx), 'heldout_crops': int(len(real['y'])),
			'generated': int(len(gen['y'])),
			'per_class': {name: {'bank': int(np.sum(bank['y'] == c)),
				'heldout': int(np.sum(real['y'] == c)), 'generated': int(np.sum(gen['y'] == c))}
				for c, name in enumerate(classes)},
		},
		**report,
	}
	write_outputs(report, arrays, bank, real, gen, classes, out)
	for name in ('patterns75', 'groups11'):
		for s in ('real', 'synthetic'):
			m = report[name][s]
			print(f"{name:10s} {s:9s} acc {m['accuracy']:.3f}  bal.acc {m['balanced_accuracy']}  "
				f"macro-F1 {m['macro_f1']}")
	d = report['distance_matrices']['patterns75']
	print(f"diagonal is row minimum: generated {d['generated_diagonal_min_count']}/{len(classes)}, "
		f"baseline {d['baseline_diagonal_min_count']}/{len(classes)}")
	print(f'results -> {out}')


if __name__ == '__main__':
	main()
