'''
Tests of ordinal.py and of the torch-free part of ordinal_analysis.py.

Run from conditional/:   python test_ordinal.py      (or: pytest test_ordinal.py)

1. two_by_two_distribution against ordpy.two_by_two_patterns(tie_patterns=True)
   if ordpy is installed, and always against two independent numpy references:
   the reference code of the method specification and a re-implementation of
   ordpy's own algorithm (argsort + dense ranks, rank_sort_ties).
2. permutation_distribution / entropy_complexity against the reference code,
   against ordpy.complexity_entropy (if installed) and the ordpy docstring values.
3. Hellinger 1-NN against brute force; classification scores against sklearn
   (if installed) or hand-computed values.
4. Timing of two_by_two_distribution on a 256 x 256 image.
5. The whole analysis (features, 1-NN, distance matrices, JSON, figures) on
   synthetic "phases" made of smoothed noise and stripes.
'''
import itertools
import json
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

import ordinal
from ordinal import (PATTERNS, GROUPS, to_gray, two_by_two_distribution, group_distribution,
	permutation_distribution, entropy_complexity, hellinger, nearest_neighbour)
import ordinal_analysis as oa

try:
	import ordpy
except ImportError:
	ordpy = None


# ---------------------------------------------------------------- references

def ref_hist75(X):
	'''Reference code of the specification (Section 7).'''
	X = np.asarray(X, np.float64)
	W = np.stack([X[:-1, :-1].ravel(), X[:-1, 1:].ravel(), X[1:, :-1].ravel(), X[1:, 1:].ravel()], axis=1)
	less = W[:, None, :] < W[:, :, None]
	eq_prev = np.zeros_like(W, dtype=bool)
	for j in range(1, 4):
		eq_prev[:, j] = (W[:, :j] == W[:, j:j+1]).any(axis=1)
	R = (less & ~eq_prev[:, None, :]).sum(axis=2)
	lut = np.full(256, -1, int)
	for i, p in enumerate(PATTERNS):
		lut[int(p, 4)] = i
	idx = lut[R @ np.array([64, 16, 4, 1])]
	return np.bincount(idx, minlength=75) / idx.size


def ordpy_style_hist75(X):
	'''ordpy's algorithm: sliding windows, argsort, dense rank = cumsum of "new value" flags.'''
	X = np.asarray(X, np.float64)
	W = sliding_window_view(X, (2, 2)).reshape(-1, 4)
	ranks = []
	for w in W:
		order = np.argsort(w)
		a = w[order]
		obs = np.r_[True, a[1:] != a[:-1]]
		inv = np.empty(4, int)
		inv[order] = np.arange(4)
		ranks.append(''.join(map(str, np.cumsum(obs)[inv] - 1)))
	return np.array([ranks.count(p) for p in PATTERNS]) / len(ranks)


def ref_entropy_complexity(X, dx, dy):
	'''Reference code of the specification (bp_distribution + entropy_complexity).'''
	n = math.factorial(dx * dy)
	P = sliding_window_view(np.asarray(X, float), (dy, dx))
	S = np.argsort(P.reshape(-1, dx * dy), axis=1, kind='stable')
	_, c = np.unique(S, axis=0, return_counts=True)
	p = np.concatenate([c / c.sum(), np.zeros(n - len(c))])
	ent = lambda q: -np.sum(q[q > 0] * np.log(q[q > 0]))
	H = ent(p) / np.log(n)
	D = ent((p + 1/n) / 2) - ent(p) / 2 - np.log(n) / 2
	Dmax = -0.5 * (((n+1)/n) * np.log(n+1) + np.log(n) - 2*np.log(2*n))
	return H, H * D / Dmax


def sample_images(rng):
	'''Images with and without ties.'''
	yield rng.random((40, 37))
	for top in (1, 2, 3, 5):
		yield rng.integers(0, top + 1, size=(31, 29))
	yield np.zeros((8, 8))
	yield to_gray(rng.integers(0, 4, size=(30, 30, 3), dtype=np.uint8))   # few colours -> ties
	yield to_gray(rng.integers(0, 256, size=(30, 30, 3), dtype=np.uint8))


# ---------------------------------------------------------------- tests

def test_tables():
	assert len(PATTERNS) == 75 and len(set(PATTERNS)) == 75
	assert [len(g) for g in GROUPS.values()] == [1, 8, 8, 4, 8, 4, 16, 8, 8, 8, 2]
	assert np.sum(ordinal.PATTERN_LOOKUP >= 0) == 75
	# all weak orderings of 4 elements are present
	weak = {''.join(str(len({v for v in r if v < x})) for x in r)
		for r in itertools.product(range(4), repeat=4)}
	assert weak == set(PATTERNS)


def test_docstring_examples():
	p = two_by_two_distribution([[1, 2, 1, 4], [8, 3, 4, 5], [6, 7, 5, 6]])
	expected = {'0132': 1/3, '1023': 1/6, '3012': 1/6, '0112': 1/3}
	for k, name in enumerate(PATTERNS):
		assert abs(p[k] - expected.get(name, 0)) < 1e-15
	g = dict(zip(ordinal.GROUP_NAMES, group_distribution(p)))
	assert abs(g['C'] - 0.5) < 1e-15 and abs(g['E'] - 1/6) < 1e-15 and abs(g['F'] - 1/3) < 1e-15
	for window, pattern in (((1, 2, 4, 2), '0121'), ((3, 1, 6, 3), '1021'),
			((1, 1, 3, 4), '0012'), ((5, 5, 5, 5), '0000')):
		p = two_by_two_distribution(np.array(window).reshape(2, 2))
		assert PATTERNS[int(np.argmax(p))] == pattern and p.max() == 1
	H, C = entropy_complexity(permutation_distribution([[1, 2, 1], [8, 3, 4], [6, 7, 5]], 2, 2))
	assert abs(H - 0.3271564379782973) < 1e-12 and abs(C - 0.2701200547320647) < 1e-12
	H, C = entropy_complexity(permutation_distribution([[1, 2, 1, 4], [8, 3, 4, 5], [6, 7, 5, 6]], 3, 2))
	assert abs(H - 0.21070701155008006) < 1e-12 and abs(C - 0.20704765093242872) < 1e-12


def test_two_by_two_against_references():
	rng = np.random.default_rng(0)
	for X in sample_images(rng):
		p = two_by_two_distribution(X)
		assert np.array_equal(p, ref_hist75(X)), 'differs from spec reference'
		assert np.allclose(p, ordpy_style_hist75(X), atol=1e-15), 'differs from ordpy-style code'
		if ordpy is not None:
			d = ordpy.two_by_two_patterns(X, taux=1, tauy=1, overlapping=True,
				tie_patterns=True, group_patterns=False)
			q = np.array([d[f'[{name}]'] for name in PATTERNS])
			assert np.allclose(p, q, atol=1e-15), 'differs from ordpy'
			dg = ordpy.two_by_two_patterns(X, tie_patterns=True, group_patterns=True)
			assert np.allclose(group_distribution(p), [dg[g] for g in ordinal.GROUP_NAMES], atol=1e-15)


def test_entropy_complexity_against_references():
	rng = np.random.default_rng(1)
	for X in sample_images(rng):
		for dx, dy in ((2, 2), (3, 2), (2, 3)):
			if X.shape[0] < dy or X.shape[1] < dx:
				continue
			p = permutation_distribution(X, dx, dy)
			assert len(p) == math.factorial(dx * dy) and abs(p.sum() - 1) < 1e-12
			H, C = entropy_complexity(p)
			Hr, Cr = ref_entropy_complexity(X, dx, dy)
			assert abs(H - Hr) < 1e-12 and abs(C - Cr) < 1e-12
			if ordpy is not None:
				Ho, Co = ordpy.complexity_entropy(X, dx=dx, dy=dy)
				assert abs(H - Ho) < 1e-12 and abs(C - Co) < 1e-12


def test_gray():
	rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [10, 20, 30]]], np.uint8)
	g = to_gray(rgb)
	assert g.dtype == np.float64
	assert np.allclose(g[0], [0.2125*255, 0.7154*255, 0.0721*255, 0.2125*10 + 0.7154*20 + 0.0721*30])


def test_nearest_neighbour():
	rng = np.random.default_rng(2)
	bank = rng.dirichlet(np.ones(75), size=500)
	labels = rng.integers(0, 7, size=500)
	queries = rng.dirichlet(np.ones(75), size=300)
	pred, dist = nearest_neighbour(bank, labels, queries, chunk=64)
	full = hellinger(queries[:, None, :], bank[None, :, :])
	assert np.array_equal(pred, labels[np.argmin(full, axis=1)])
	assert np.allclose(dist, full.min(axis=1), atol=1e-7)
	_, d0 = nearest_neighbour(bank, labels, bank[:5])
	assert np.all(d0 < 1e-6)


def test_classification_metrics():
	y = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
	p = np.array([0, 1, 1, 1, 0, 2, 2, 1, 2])
	counts, matrix, m = oa.classification_metrics(y, p, 3)
	assert counts.tolist() == [[1, 1, 0], [1, 2, 0], [0, 1, 3]]
	assert abs(m['accuracy'] - 6/9) < 1e-12
	assert abs(m['balanced_accuracy'] - (1/2 + 2/3 + 3/4) / 3) < 1e-12
	try:
		from sklearn.metrics import f1_score, balanced_accuracy_score
		assert abs(m['macro_f1'] - f1_score(y, p, average='macro')) < 1e-12
		assert abs(m['balanced_accuracy'] - balanced_accuracy_score(y, p)) < 1e-12
	except ImportError:
		f1 = [2*1/(2*1+1+1), 2*2/(2*2+2+1), 2*3/(2*3+0+1)]
		assert abs(m['macro_f1'] - np.mean(f1)) < 1e-12
	assert np.allclose(oa.normalized_hundredths(np.array([[1, 1, 1]])).sum(), 1)


def test_timing():
	rng = np.random.default_rng(3)
	gray = to_gray(rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8))
	two_by_two_distribution(gray)
	n = 50
	t = time.perf_counter()
	for _ in range(n):
		two_by_two_distribution(gray)
	ms = 1000 * (time.perf_counter() - t) / n
	t = time.perf_counter()
	for _ in range(n):
		oa.image_features(rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8))
	ms_all = 1000 * (time.perf_counter() - t) / n
	print(f'  two_by_two_distribution 256x256: {ms:.1f} ms; all features per crop: {ms_all:.1f} ms')
	assert ms < 50


# ---------------------------------------------------------------- end-to-end

def texture(kind, rng, size=300):
	'''Synthetic "phase": smoothed noise of a given scale, or stripes, as uint8 RGB.'''
	from numpy.fft import fft2, ifft2, fftfreq
	k = np.hypot(*np.meshgrid(fftfreq(size), fftfreq(size)))
	noise = rng.standard_normal((size, size))
	if kind[0] == 'smooth':
		field = np.real(ifft2(fft2(noise) * np.exp(-(k * kind[1])**2)))
	else:
		y, x = np.mgrid[:size, :size]
		angle = kind[1] + 0.1 * rng.standard_normal()
		field = np.sin((x * np.cos(angle) + y * np.sin(angle)) / 4) + 0.3 * noise
	field = (field - field.min()) / (np.ptp(field) + 1e-12)
	levels = kind[2]                       # few grey levels -> many ties, like saturated POM
	g = np.round(field * (levels - 1)) / (levels - 1)
	rgb = np.stack([g, g**1.5, g**0.7], axis=-1)
	return (255 * rgb).astype(np.uint8)


KINDS = [('smooth', 2, 256), ('smooth', 8, 256), ('smooth', 30, 256), ('smooth', 8, 6),
	('stripes', 0.0, 256), ('stripes', 1.2, 256), ('smooth', 2, 4)]
CLASSES = ['N', 'SmA', 'SmB', 'SmC', 'SmF', 'SmG', 'SmI']


def test_end_to_end():
	rng = np.random.default_rng(4)
	with tempfile.TemporaryDirectory() as tmp:
		tmp = Path(tmp)
		paths = {'train': [], 'test': []}
		labels = {'train': [], 'test': []}
		for split, count in (('train', 6), ('test', 3)):
			for c, kind in enumerate(KINDS):
				for i in range(count):
					p = tmp / split / CLASSES[c] / f'{i}.png'
					p.parent.mkdir(parents=True, exist_ok=True)
					Image.fromarray(texture(kind, rng)).save(p)
					paths[split].append(str(p))
					labels[split].append(c)
		gen_dir = tmp / 'generated'
		for c, kind in enumerate(KINDS):
			(gen_dir / CLASSES[c]).mkdir(parents=True)
			for i in range(5):
				Image.fromarray(texture(kind, rng, 256)).save(gen_dir / CLASSES[c] / f'{i:05d}.png')

		bank = oa.feature_table(paths['train'], labels['train'], 256, workers=2)
		real = oa.feature_table(paths['test'], labels['test'], 256, workers=1)
		gen = oa.feature_table(*oa.generated_files(gen_dir, CLASSES), None, workers=2)
		assert bank['p75'].shape == (4 * 6 * 7, 75) and real['p75'].shape == (4 * 3 * 7, 75)
		assert gen['p11'].shape == (5 * 7, 11) and np.allclose(gen['p11'].sum(1), 1)

		report, arrays = oa.analyse(bank, real, gen, CLASSES)
		out = tmp / 'ordinal'
		oa.write_outputs(report, arrays, bank, real, gen, CLASSES, out)
		r = json.loads((out / 'ordinal_metrics.json').read_text())
		print('  synthetic phases: 75-pattern bal. acc real %.3f, generated %.3f; 11-group %.3f / %.3f; '
			'diagonal minimum %d/7' % (r['patterns75']['real']['balanced_accuracy'],
			r['patterns75']['synthetic']['balanced_accuracy'],
			r['groups11']['real']['balanced_accuracy'], r['groups11']['synthetic']['balanced_accuracy'],
			r['distance_matrices']['patterns75']['generated_diagonal_min_count']))
		assert r['patterns75']['real']['balanced_accuracy'] > 0.9
		assert r['patterns75']['synthetic']['balanced_accuracy'] > 0.9
		assert r['distance_matrices']['patterns75']['generated_diagonal_min_count'] >= 6
		for name in ('ordinal_confusion', 'ordinal_confusion_groups', 'hc_plane', 'ordinal_distance'):
			for ext in ('pdf', 'png'):
				assert (out / f'{name}.{ext}').stat().st_size > 1000, name
		keep = os.environ.get('ORDINAL_TEST_OUT')   # optional: folder to keep the figures
		if keep:
			import shutil
			Path(keep).mkdir(parents=True, exist_ok=True)
			for f in out.glob('*.p*'):
				shutil.copy(f, Path(keep) / f.name)


if __name__ == '__main__':
	print('ordpy installed:', ordpy is not None)
	for name, f in list(globals().items()):
		if name.startswith('test_') and callable(f):
			f()
			print('ok ', name)
