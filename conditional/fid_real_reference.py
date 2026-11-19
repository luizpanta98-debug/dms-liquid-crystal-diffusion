'''
Reference FID and KID between two disjoint sets of real textures.

The images of every class are split at random into two halves (A, B), so no
image contributes to both sets. From half A we draw N_A and from half B N_B
distinct crop/flip views (four corner crops x four flips per image) with the
class proportions of the dataset, exactly as for plot_fid.py, and compute
FID(A, B) and KID(A, B). With N_A = generated images and N_B = real reference
of plot_fid.py, this is the value of a "perfect" generator under the same
protocol: the floor against which the model is compared.

No network weights are loaded; only the pytorch-fid Inception network is used.

Usage (from conditional/):
    python fid_real_reference.py                     # N_A=10000, N_B=50000
    python fid_real_reference.py --pairs 10000:50000 50000:50000
Output: <save_checkpoint>/metrics/fid_real_reference.json (new pairs are added
to the results already in the file).
'''
import os
import json
import argparse
import numpy as np
import torch as th

from pytorch_fid.fid_score import calculate_frechet_distance
from plot_fid import (
	CropDataset, real_dataset_and_counts, sample_labels, get_inception_model,
	real_features, stats_of, kid, GEN_SAMPLES, NUM_SAMPLES)


def split_halves(targets, num_classes, rng):
	'''Image indices of two disjoint halves, split within each class.'''
	half_a, half_b = [], []
	for c in range(num_classes):
		idx = rng.permutation(np.flatnonzero(targets == c))
		half_a.append(idx[:len(idx)//2])
		half_b.append(idx[len(idx)//2:])
	return np.concatenate(half_a), np.concatenate(half_b)


def views_of(images, targets, class_counts, rng):
	'''Distinct crop/flip views (16 per image), class_counts views per class.'''
	chosen = []
	for c, n in enumerate(class_counts):
		sources = images[targets[images] == c]
		views = (sources[:, None]*16 + np.arange(16)).ravel()
		if n > len(views):
			raise ValueError(f'class {c}: {n} views requested, {len(views)} available')
		chosen.append(rng.choice(views, size=int(n), replace=False))
	chosen = np.concatenate(chosen)
	rng.shuffle(chosen)
	return chosen


def main():
	p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
	p.add_argument('--pairs', nargs='+', default=[f'{GEN_SAMPLES}:{NUM_SAMPLES}'],
		help='N_A:N_B, views of half A and of half B (one FID and KID per pair)')
	args = p.parse_args()
	pairs = [tuple(int(n) for n in pair.split(':')) for pair in args.pairs]

	with open('config.json') as f:
		config = json.load(f)
	device = config['device']
	rng = np.random.RandomState(config['seed'])

	dataset, counts = real_dataset_and_counts(config)
	targets = np.array(dataset.targets)
	half_a, half_b = split_halves(targets, config['label_dim'], rng)
	print(f'images: {len(dataset)} (half A {len(half_a)}, half B {len(half_b)})')

	out_dir = os.path.join(config['save_checkpoint'], 'metrics')
	os.makedirs(out_dir, exist_ok=True)
	out_json = os.path.join(out_dir, 'fid_real_reference.json')
	results = []
	if os.path.isfile(out_json):
		with open(out_json) as f:
			results = json.load(f).get('results', [])

	model = get_inception_model(device)
	for num_a, num_b in pairs:
		feats = []
		for name, half, num in (('A', half_a, num_a), ('B', half_b, num_b)):
			class_counts = sample_labels(counts, config['seed'], num)[1]
			views = views_of(half, targets, class_counts, rng)
			subset = CropDataset(dataset, views, config['img_resolution'])
			with th.inference_mode():
				feats.append(real_features(subset, device, model))
			print(f'{num_a}:{num_b}: features of half {name} done')
		(mu_a, sig_a), (mu_b, sig_b) = stats_of(feats[0]), stats_of(feats[1])
		fid = float(calculate_frechet_distance(mu_a, sig_a, mu_b, sig_b))
		kid_mean, kid_std = kid(feats[0], feats[1], seed=config['seed'])
		print(f'real A ({num_a}) vs real B ({num_b}): FID={fid:.3f}  KID={kid_mean:.5f} +- {kid_std:.5f}')
		results = [r for r in results
			if (r.get('num_a', r.get('num_per_set')), r.get('num_b', r.get('num_per_set'))) != (num_a, num_b)]
		results.append({'num_a': num_a, 'num_b': num_b, 'fid': fid,
			'kid': kid_mean, 'kid_std_over_subsets': kid_std})

		report = {
			'metric': 'FID and KID between two disjoint halves of the real images (split by image within each class)',
			'views': 'corner crops x flips, class proportions of the dataset, without replacement',
			'seed': config['seed'],
			'images_half_a': int(len(half_a)), 'images_half_b': int(len(half_b)),
			'results': results,
		}
		with open(out_json, 'w') as f:
			json.dump(report, f, indent=4)
		print(f'saved -> {out_json}')


if __name__ == '__main__':
	main()
