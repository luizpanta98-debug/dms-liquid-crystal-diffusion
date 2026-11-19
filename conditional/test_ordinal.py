'''
Tests of ordinal.py and of the torch-free part of ordinal_analysis.py.

Run from conditional/:   python test_ordinal.py      (or: pytest test_ordinal.py)

1. permutation_distribution / entropy_complexity against an independent numpy
   reference, against ordpy.complexity_entropy (if installed) and the ordpy
   docstring values.
2. Luminance conversion.
3. The whole analysis (features, summary, JSON, figure) on synthetic "phases"
   made of smoothed noise and stripes.
'''
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

from ordinal import to_gray, permutation_distribution, entropy_complexity
import ordinal_analysis as oa

try:
	import ordpy
except ImportError:
	ordpy = None


# ---------------------------------------------------------------- references

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

def test_docstring_examples():
	H, C = entropy_complexity(permutation_distribution([[1, 2, 1], [8, 3, 4], [6, 7, 5]], 2, 2))
	assert abs(H - 0.3271564379782973) < 1e-12 and abs(C - 0.2701200547320647) < 1e-12
	H, C = entropy_complexity(permutation_distribution([[1, 2, 1, 4], [8, 3, 4, 5], [6, 7, 5, 6]], 3, 2))
	assert abs(H - 0.21070701155008006) < 1e-12 and abs(C - 0.20704765093242872) < 1e-12


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
		paths, labels = [], []
		for c, kind in enumerate(KINDS):
			for i in range(3):
				p = tmp / 'test' / CLASSES[c] / f'{i}.png'
				p.parent.mkdir(parents=True, exist_ok=True)
				Image.fromarray(texture(kind, rng)).save(p)
				paths.append(str(p))
				labels.append(c)
		gen_dir = tmp / 'generated'
		for c, kind in enumerate(KINDS):
			(gen_dir / CLASSES[c]).mkdir(parents=True)
			for i in range(5):
				Image.fromarray(texture(kind, rng, 256)).save(gen_dir / CLASSES[c] / f'{i:05d}.png')

		real = oa.feature_table(paths, labels, 256, workers=2)
		gen = oa.feature_table(*oa.generated_files(gen_dir, CLASSES), None, workers=1)
		assert real['hc'].shape == (4 * 3 * 7, 2) and gen['hc'].shape == (5 * 7, 2)
		assert np.all((real['hc'] >= 0) & (real['hc'] <= 1))

		report = {'hc_plane': {'real_heldout': oa.hc_summary(real['hc'], real['y'], CLASSES),
			'synthetic': oa.hc_summary(gen['hc'], gen['y'], CLASSES)}}
		out = tmp / 'ordinal'
		oa.write_outputs(report, real, gen, CLASSES, out)
		r = json.loads((out / 'ordinal_metrics.json').read_text())
		# The same synthetic texture gives the same (H, C) for "real" and "generated".
		for name in CLASSES:
			a, b = r['hc_plane']['real_heldout'][name], r['hc_plane']['synthetic'][name]
			assert abs(a['H_mean'] - b['H_mean']) < 0.05, name
		for ext in ('pdf', 'png'):
			assert (out / f'hc_plane.{ext}').stat().st_size > 1000
		keep = os.environ.get('ORDINAL_TEST_OUT')   # optional: folder to keep the figure
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
