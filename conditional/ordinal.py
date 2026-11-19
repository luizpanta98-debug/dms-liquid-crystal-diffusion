'''
Complexity-entropy plane of ordinal patterns (pure numpy, no torch).

The image is converted to luminance Y = 0.2125 R + 0.7154 G + 0.0721 B
(float64, never rounded back to uint8). Each dy x dx window of neighbouring
pixels is replaced by the permutation that sorts it (Bandt-Pompe), with ties
broken by order of occurrence (stable argsort). From the (dx dy)! pattern
frequencies p we compute the normalised permutation entropy H and the
Jensen-Shannon statistical complexity C (Ribeiro et al. 2012; Sigaki et al.,
PRE 99, 013311, 2019).

Usage:
	from ordinal import to_gray, permutation_distribution, entropy_complexity
	gray = to_gray(rgb_uint8)                 # (H, W, 3) uint8 -> (H, W) float64
	H, C = entropy_complexity(permutation_distribution(gray, dx=2, dy=2))
'''
import itertools
import math

import numpy as np


def to_gray(rgb_uint8):
	'''uint8 RGB image (H, W, 3) -> float64 luminance (scikit-image weights).

	Do not round the result: rounding creates extra ties. (Dividing by 255, as
	rgb2gray does, would not change any pattern, so we skip it.)
	'''
	rgb = np.asarray(rgb_uint8)
	return rgb[..., :3].astype(np.float64) @ np.array([0.2125, 0.7154, 0.0721])


def permutation_distribution(gray, dx=2, dy=2):
	'''Strict Bandt-Pompe distribution of dy x dx windows ((dx dy)! entries).

	dx is the horizontal size (columns), dy the vertical one (rows), as in ordpy.
	Each window is flattened row by row and replaced by argsort(window); equal
	values keep their order of occurrence (stable sort). Unobserved patterns
	are included with probability 0; entries follow itertools.permutations order.
	'''
	from numpy.lib.stride_tricks import sliding_window_view
	n = dx * dy
	if n > 6:
		raise ValueError('dx * dy > 6 is not supported (lookup table too large)')
	X = np.asarray(gray, dtype=np.float64)
	windows = sliding_window_view(X, (dy, dx)).reshape(-1, n)
	symbols = np.argsort(windows, axis=1, kind='stable')
	codes = symbols @ (n ** np.arange(n - 1, -1, -1))           # base-n number
	lut = np.full(n**n, -1, dtype=np.int64)
	for k, perm in enumerate(itertools.permutations(range(n))):
		lut[np.dot(perm, n ** np.arange(n - 1, -1, -1))] = k
	counts = np.bincount(lut[codes], minlength=math.factorial(n))
	return counts / counts.sum()


def shannon(p):
	'''Shannon entropy (natural log) with 0 ln 0 = 0.'''
	p = np.asarray(p, dtype=np.float64)
	p = p[p > 0]
	return -np.sum(p * np.log(p))


def entropy_complexity(p):
	'''Normalised permutation entropy H and statistical complexity C of p.

	p must contain all n possible patterns (zeros included). With u = 1/n:
		H     = S(p) / ln n
		D_JS  = S((p+u)/2) - S(p)/2 - S(u)/2
		D_max = -1/2 [ (n+1)/n ln(n+1) + ln n - 2 ln(2n) ]   (D_JS at a delta)
		C     = H D_JS / D_max
	'''
	p = np.asarray(p, dtype=np.float64)
	n = len(p)
	H = shannon(p) / np.log(n)
	D = shannon((p + 1/n) / 2) - shannon(p) / 2 - np.log(n) / 2
	D_max = -0.5 * ((n + 1) / n * np.log(n + 1) + np.log(n) - 2 * np.log(2 * n))
	return H, H * D / D_max
