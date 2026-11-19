'''
Article figure from nearest_neighbours.json: median distance to the closest
training crop, per phase, for generated images and real held-out crops.

Run from conditional/:
    python plot_nn_summary.py logs/lq256snr_cfg/metrics/epoch_100/nearest_neighbours.json
Writes nn_summary.pdf/.png next to the JSON.
'''
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GEN, REAL = '#eb6834', '#2a78d6'           # validated categorical pair (CVD-safe)
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3df'


def main(path):
	path = Path(path)
	r = json.loads(path.read_text())
	classes = list(r['per_class'])
	rows = classes + ['All']
	gen = [r['per_class'][c]['generated_median'] for c in classes] + [r['generated_median']]
	real = [r['per_class'][c]['held_out_median'] for c in classes] + [r['held_out_median']]

	plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.edgecolor': MUTED,
		'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': INK, 'pdf.fonttype': 42})
	fig, ax = plt.subplots(figsize=(3.4, 2.55))
	y = list(range(len(rows)))[::-1]
	for yi, g, h in zip(y, gen, real):
		ax.plot([h, g], [yi, yi], color=GRID, lw=2, zorder=1, solid_capstyle='round')
		ax.text(g + 0.008, yi, f'{g/h:.1f}×', va='center', fontsize=7, color=MUTED)
	ax.scatter(real, y, s=36, color=REAL, edgecolor='white', lw=1, zorder=3, label='Held-out real crops')
	ax.scatter(gen, y, s=36, color=GEN, edgecolor='white', lw=1, zorder=3, marker='D', label='Generated')
	ax.axhline(0.5, color=GRID, lw=0.8)
	ax.set_yticks(y)
	ax.set_yticklabels(rows)
	ax.get_yticklabels()[-1].set_fontweight('bold')
	ax.set_xlim(0, max(gen) * 1.22)
	ax.set_xlabel('Median cosine distance to nearest training crop')
	ax.xaxis.grid(True, color=GRID, lw=0.6)
	ax.set_axisbelow(True)
	for s in ('top', 'right', 'left'):
		ax.spines[s].set_visible(False)
	ax.tick_params(axis='y', length=0)
	ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.45, 1.0), ncol=2,
		handletextpad=0.3, columnspacing=1.2, fontsize=7)
	fig.tight_layout(pad=0.3)
	for ext in ('pdf', 'png'):
		fig.savefig(path.with_name(f'nn_summary.{ext}'), dpi=300)
	print(f'saved -> {path.with_name("nn_summary.pdf")}')


if __name__ == '__main__':
	main(sys.argv[1] if len(sys.argv) > 1 else 'nearest_neighbours.json')
