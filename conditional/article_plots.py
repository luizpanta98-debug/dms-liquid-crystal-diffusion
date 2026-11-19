'''
Article plot style, shared by all scripts that draw figures for the article.

plot_confusion(folder, out, layout): ResNet-50 confusion matrices, real (a) and
generated (b), beside each other (layout='row', as in the article) or one above
the other ('column'); Blues, every cell annotated with sum-preserving rounded
proportions. <folder> must contain confusion_matrices.npz in the format written
by resnet50/conf_matrix.py. Used by collect_results.py.

Command line (from conditional/):
    python article_plots.py <folder with confusion_matrices.npz> out.pdf [row|column]
'''
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter


def plot_font():
	# Same choice as resnet50/conf_matrix.py.
	for name in ('Times New Roman', 'Arial', 'DejaVu Serif'):
		try:
			font_manager.findfont(font_manager.FontProperties(family=[name]), fallback_to_default=False)
			return name
		except ValueError:
			continue
	return 'DejaVu Serif'


def plot_confusion(folder, out, layout='column'):
	'''layout: 'column' (a above b) or 'row' (a beside b, for a full-width figure).'''
	d = np.load(Path(folder) / 'confusion_matrices.npz')
	classes = [str(c) for c in d['classes']]
	n = len(classes)
	panels = (
		(d['real_display_normalized'], int(d['real_counts'].sum()), '(a) Real', 'True class'),
		(d['synthetic_display_normalized'], int(d['synthetic_counts'].sum()), '(b) Generated', 'Requested class'),
	)
	style = {'font.family': plot_font(), 'font.size': 10, 'axes.labelsize': 11,
		'axes.titlesize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
		'pdf.fonttype': 42, 'ps.fonttype': 42}

	with plt.rc_context(style):
		cmap = plt.get_cmap('Blues').with_extremes(bad='#EEEEEE')
		if layout == 'row':
			fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3), constrained_layout=True)
		else:
			fig, axes = plt.subplots(2, 1, figsize=(4.6, 7.5), constrained_layout=True)
		for k, (ax, (matrix, total, title, ylabel)) in enumerate(zip(axes, panels)):
			image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=1)
			ax.set(title=f'{title} (N={total})', ylabel=ylabel,
				xticks=np.arange(n), yticks=np.arange(n))
			ax.set_yticklabels(classes)
			if layout == 'row' or k == len(panels) - 1:
				ax.set_xlabel('Predicted class')
				ax.set_xticklabels(classes, rotation=45, ha='right', rotation_mode='anchor')
			else:
				ax.set_xticklabels([])
			ax.tick_params(which='both', length=0)
			ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
			ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
			ax.grid(which='minor', color='white', linewidth=0.6)
			for spine in ax.spines.values():
				spine.set_visible(False)
			for i in range(n):
				for j in range(n):
					v = matrix[i, j]
					ax.text(j, i, f'{v:.2f}' if np.isfinite(v) else '—',
						ha='center', va='center', fontsize=10,
						color='white' if v > 0.6 else 'black')

		fig.colorbar(image, ax=axes, ticks=np.linspace(0, 1, 6),
			format=FormatStrFormatter('%.2f'), label='Row-normalised proportion',
			shrink=0.6, aspect=30, pad=0.03)
		fig.savefig(out, bbox_inches='tight', pad_inches=0.06)
		fig.savefig(str(out).rsplit('.', 1)[0] + '.png', dpi=300, bbox_inches='tight', pad_inches=0.06)
	print(f'saved -> {out}')


if __name__ == '__main__':
	# optional third argument: row (panels side by side) or column (default)
	plot_confusion(sys.argv[1], sys.argv[2], *(sys.argv[3:4]))
