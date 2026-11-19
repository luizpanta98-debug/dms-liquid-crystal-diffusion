'''
Ordinal-pattern texture descriptors (pure numpy, no torch).

Two families of descriptors, both computed on the grayscale image
Y = 0.2125 R + 0.7154 G + 0.0721 B (float64, never rounded back to uint8):

1. Two-by-two patterns WITH ties (Tarozo et al. 2025; Voltarelli et al. 2026,
   arXiv:2603.26723). Every overlapping 2x2 window (a0 a1 / a2 a3), flattened
   row by row, is replaced by the dense ranks of its four values: equal values
   share a rank, ranks are consecutive. Example: (1,2,4,2) -> 0121. There are
   75 such weak orderings; their relative frequencies form a 75-dim
   distribution, which can be summed into 11 groups A-K. The order of the 75
   patterns is the order used by ordpy.two_by_two_patterns(tie_patterns=True).

2. Strict Bandt-Pompe patterns (Ribeiro et al. 2012; Sigaki et al. PRE 2019).
   Each dy x dx window is replaced by the permutation that sorts it, with ties
   broken by occurrence order (stable argsort). From the (dx dy)! pattern
   frequencies p we compute the normalised permutation entropy H and the
   Jensen-Shannon statistical complexity C.

Distributions are compared with the Hellinger distance
	d_H(p, q) = sqrt( 1/2 sum_k (sqrt p_k - sqrt q_k)^2 ),
i.e. the Euclidean distance between sqrt(p) and sqrt(q), up to 1/sqrt(2).

Usage:
	from ordinal import to_gray, two_by_two_distribution, group_distribution
	gray = to_gray(rgb_uint8)                 # (H, W, 3) uint8 -> (H, W) float64
	p75 = two_by_two_distribution(gray)       # 75 probabilities
	p11 = group_distribution(p75)             # 11 probabilities, groups A..K
	H, C = entropy_complexity(permutation_distribution(gray, dx=2, dy=2))
'''
import itertools
import math

import numpy as np


# ---------------------------------------------------------------- pattern tables

# The 75 tie-aware 2x2 patterns, grouped A-K, copied from the GROUPS literal of
# ordpy.two_by_two_patterns. Each string is the rank vector of (a0, a1, a2, a3).
GROUPS = {
	'A': ['0000'],
	'B': ['0123', '2031', '3210', '1302', '0213', '1032', '3120', '2301'],
	'C': ['0132', '3021', '2310', '1203', '0312', '1023', '2130', '3201'],
	'D': ['0011', '1010', '1100', '0101'],
	'E': ['0231', '3012', '1320', '2103', '0321', '2013', '1230', '3102'],
	'F': ['0112', '1021', '2110', '1201'],
	'G': ['0012', '1020', '2100', '0201', '0021', '2010', '1200', '0102',
		'0122', '2021', '2210', '1202', '0212', '1022', '2120', '2201'],
	'H': ['0001', '0010', '1000', '0100', '0111', '1011', '1110', '1101'],
	'I': ['0121', '2011', '1210', '1102', '0211', '1012', '1120', '2101'],
	'J': ['0120', '2001', '0210', '1002', '0221', '2012', '1220', '2102'],
	'K': ['0110', '1001'],
}
GROUP_NAMES = list(GROUPS)
PATTERNS = [p for g in GROUPS.values() for p in g]          # 75, in ordpy order
GROUP_OF_PATTERN = np.array([k for k, g in enumerate(GROUPS.values()) for _ in g])

# The six pairs (i, j) of positions inside a window, i < j.
PAIRS = list(itertools.combinations(range(4), 2))


def comparison_key(values):
	'''Encode the six pairwise comparisons of four values as one integer.

	A weak ordering of four values (which value is smaller, equal, larger) is
	fully determined by the comparisons of the six pairs. Each comparison is
	0 (<), 1 (=) or 2 (>), so the key is a base-3 number in 0 .. 3^6 - 1 = 728.
	values: sequence of 4 arrays (or numbers) of the same shape.
	'''
	key = 0
	for i, j in PAIRS:
		sign = (values[i] > values[j]).astype(np.int16) - (values[i] < values[j])
		key = 3 * key + (sign + 1)
	return key


def _build_lookup():
	'''Table: comparison key (0..728) -> index of the pattern in PATTERNS (or -1).

	Only 75 of the 729 keys are consistent orderings; the others never occur.
	'''
	lut = np.full(3**6, -1, dtype=np.int64)
	for k, pattern in enumerate(PATTERNS):
		ranks = [np.array(int(c)) for c in pattern]
		lut[int(comparison_key(ranks))] = k
	return lut


PATTERN_LOOKUP = _build_lookup()


# ---------------------------------------------------------------- descriptors

def to_gray(rgb_uint8):
	'''uint8 RGB image (H, W, 3) -> float64 luminance (scikit-image weights).

	Do not round the result: rounding creates extra ties and changes the
	tie-pattern statistics. (Dividing by 255, as rgb2gray does, would not change
	any pattern, so we skip it.)
	'''
	rgb = np.asarray(rgb_uint8)
	return rgb[..., :3].astype(np.float64) @ np.array([0.2125, 0.7154, 0.0721])


def two_by_two_distribution(gray):
	'''75-dim distribution of tie-aware 2x2 ordinal patterns (ordpy order).

	Windows: all overlapping 2x2 blocks with unit step,
	(a0, a1, a2, a3) = (X[r, c], X[r, c+1], X[r+1, c], X[r+1, c+1]).
	'''
	X = np.asarray(gray, dtype=np.float64)
	windows = (X[:-1, :-1], X[:-1, 1:], X[1:, :-1], X[1:, 1:])
	key_counts = np.bincount(comparison_key(windows).ravel(), minlength=3**6)
	observed = np.flatnonzero(key_counts)
	if (PATTERN_LOOKUP[observed] < 0).any():
		raise ValueError('inconsistent comparisons (NaN values in the image?)')
	counts = np.bincount(PATTERN_LOOKUP[observed], weights=key_counts[observed],
		minlength=len(PATTERNS))
	return counts / counts.sum()


def group_distribution(p75):
	'''Sum a 75-dim distribution (or an array of them, last axis) into groups A-K.'''
	p75 = np.asarray(p75, dtype=np.float64)
	out = np.zeros(p75.shape[:-1] + (len(GROUPS),))
	for k in range(len(GROUPS)):
		out[..., k] = p75[..., GROUP_OF_PATTERN == k].sum(axis=-1)
	return out


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


# ---------------------------------------------------------------- distances and 1-NN

def hellinger(P, Q):
	'''Hellinger distance between distributions along the last axis (broadcasts).'''
	P, Q = np.asarray(P, dtype=np.float64), np.asarray(Q, dtype=np.float64)
	return np.sqrt(0.5 * np.sum((np.sqrt(P) - np.sqrt(Q))**2, axis=-1))


def nearest_neighbour(bank, bank_labels, queries, chunk=2048):
	'''1-NN classification with the Hellinger distance.

	bank: (N, K) distributions with labels bank_labels (N,); queries: (M, K).
	Returns (predicted labels (M,), distance to the nearest bank entry (M,)).
	For normalised distributions, d_H^2 = 1 - sum_k sqrt(p_k q_k), so the
	nearest neighbour is the bank entry with the largest overlap sqrt(p).sqrt(q).
	'''
	B = np.sqrt(np.asarray(bank, dtype=np.float64))
	Q = np.sqrt(np.asarray(queries, dtype=np.float64))
	# Use the general form |a|^2 + |b|^2 - 2 a.b, so that rounding in the
	# normalisation of the inputs does not matter.
	b2 = (B**2).sum(axis=1)
	nearest, dist = [], []
	for start in range(0, len(Q), chunk):
		q = Q[start:start + chunk]
		d2 = (q**2).sum(axis=1)[:, None] + b2[None, :] - 2 * q @ B.T
		k = np.argmin(d2, axis=1)
		nearest.append(k)
		dist.append(np.sqrt(0.5 * np.clip(d2[np.arange(len(q)), k], 0, None)))
	nearest = np.concatenate(nearest) if nearest else np.zeros(0, int)
	dist = np.concatenate(dist) if dist else np.zeros(0)
	return np.asarray(bank_labels)[nearest], dist
