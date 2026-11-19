'''
Collect the evaluation outputs of one or more trained models into LaTeX files
for the article (Overleaf folder "results/").

Run from conditional/ after nll_calc.py, plot_fid.py and
resnet50/test_script.py have been executed for each model:

    python collect_results.py \
        --runs logs/lq256snr_cfg logs/lq256snr_cfg_eps \
        --names x0 eps \
        --main x0 \
        --main_steps 100 \
        --epoch 100 \
        --classifier resnet50/logs/resnet50/test \
        --out ../results

Inputs read for each run directory R (save_checkpoint of that model) and
evaluated epoch E (--epoch; "eval_epoch" in config.json when evaluating):
    R/config.json
    R/metrics/epoch_E/nll.json
    R/metrics/epoch_E/fid_and_inception.json
    R/metrics/epoch_E/nearest_neighbours.json, nn_distances.pdf, nearest_neighbours.pdf (optional)
    R/metrics/epoch_E/ordinal/ordinal_metrics.json and its figures (optional, ordinal_analysis.py)
    <classifier>/<basename of R>/epoch_E/metrics.json            (optional)
    <classifier>/<basename of R>/epoch_E/confusion_matrices.pdf  (optional, copied)

Outputs (upload the whole folder to Overleaf, replacing results/):
    macros.tex            numbers quoted in the text (\\FIDmain, \\NLLmain, ...)
    table_models.tex      NLL / FID / classifier metrics
    table_fid_steps.tex   FID against the number of DDIM steps (only with a curve)
    fid_vs_steps.pdf      FID against DDIM steps, one curve per model
    confusion_<name>.pdf  ResNet-50 confusion matrices for each model
    nn_distances_<name>.pdf, nn_pairs_<name>.pdf  nearest-neighbour check
    ordinal_confusion_<name>.pdf, hc_plane_<name>.pdf, ordinal_distance_<name>.pdf
                          ordinal-pattern comparison (ordinal_analysis.py)
'''
import argparse
import json
import shutil
from pathlib import Path


def load(path):
	path = Path(path)
	return json.loads(path.read_text()) if path.exists() else None


def fmt(value, digits):
	return '--' if value is None else f'{value:.{digits}f}'


def macro_name(name):
	# LaTeX macro names cannot contain digits or symbols: x0 -> xzero.
	table = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
		'5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'}
	return ''.join(table.get(c, c) for c in name if c.isalnum())


def pick_epoch(run, epoch):
	"""Evaluated epoch of a run: the one requested, or the only one available."""
	if epoch is not None:
		return epoch
	found = sorted(int(p.name.split('_')[1]) for p in (Path(run) / 'metrics').glob('epoch_*'))
	if len(found) != 1:
		raise SystemExit(f'{run}: found evaluated epochs {found}; choose one with --epoch')
	return found[0]


def collect(run, name, classifier_dir, epoch):
	run = Path(run)
	epoch = pick_epoch(run, epoch)
	metrics = run / 'metrics' / f'epoch_{epoch}'
	config = load(run / 'config.json') or {}
	nll = load(metrics / 'nll.json')
	fid = load(metrics / 'fid_and_inception.json')
	nn = load(metrics / 'nearest_neighbours.json')
	ordinal = load(metrics / 'ordinal' / 'ordinal_metrics.json')
	cls = None
	if classifier_dir is not None:
		cls = load(Path(classifier_dir) / run.name / f'epoch_{epoch}' / 'metrics.json')

	steps = {}
	if fid is not None:
		for r in fid['results']:
			steps[r['num_steps']] = r

	return {
		'name': name,
		'dir': run,
		'epoch': epoch,
		'metrics': metrics,
		'prediction_type': config.get('prediction_type'),
		'guidance_scale': config.get('guidance_scale'),
		'nll': None if nll is None else nll['nll_bpd'],
		'nll_count': None if nll is None else nll['num_samples'],
		'nll_se': None if nll is None else nll.get('nll_bpd_standard_error'),
		'steps': steps,
		'main_result': (fid or {}).get('main'),
		'fid_meta': fid,
		'classifier': cls,
		'nn': nn,
		'ordinal': ordinal,
	}


def main_value(r, main_steps):
	'''FID reported as main result: the large-N run if available, else the curve point.'''
	m = r.get('main_result')
	if m and m.get('num_steps') == main_steps:
		return m
	return r['steps'].get(main_steps, {})


def write_tables(runs, main, main_steps, out):
	label = {'x0': r'$x_0$', 'eps': r'$\epsilon$'}

	# Model comparison table.
	lines = [
		r'\begin{tabular}{lcccc}',
		r'\hline',
		r'Images & NLL & FID & Bal.\ acc. & Macro F1 \\',
		r'\hline',
	]
	for r in runs:
		s = main_value(r, main_steps)
		gen = (r['classifier'] or {}).get('synthetic', {})
		name = label.get(r['prediction_type'], r['name'])
		lines.append(' & '.join([
			'Generated' if len(runs) == 1 else f'Generated ({name})',
			fmt(r['nll'], 2) if r.get('nll_se') is None else f"${fmt(r['nll'], 2)}\\pm{fmt(r['nll_se'], 2)}$",
			fmt(s.get('fid'), 1),
			fmt(gen.get('balanced_accuracy'), 3),
			fmt(gen.get('macro_f1'), 3),
		]) + r' \\')
	real = next((r['classifier']['real'] for r in runs if r['classifier']), None)
	if real is not None:
		lines.append(r'\hline')
		lines.append(' & '.join([
			'Real held-out crops', '--', '--',
			fmt(real.get('balanced_accuracy'), 3),
			fmt(real.get('macro_f1'), 3)]) + r' \\')
	lines += [r'\hline', r'\end{tabular}']
	(out / 'table_models.tex').write_text('\n'.join(lines) + '\n')

	# FID against DDIM steps (only if a curve was computed).
	all_steps = sorted({k for r in runs for k in r['steps']})
	if len(all_steps) < 2:
		print('only one step count: skipping table_fid_steps.tex')
		all_steps = []
	cols = 'l' + 'c' * len(runs)
	head = ' & '.join(label.get(r['prediction_type'], r['name']) for r in runs)
	lines = [rf'\begin{{tabular}}{{{cols}}}', r'\hline',
		rf'Steps & FID ({head}) \\' if len(runs) == 1 else rf'Steps & {head} \\', r'\hline']
	for k in all_steps:
		cells = []
		for r in runs:
			s = r['steps'].get(k, {})
			cells += [fmt(s.get('fid'), 1)]
		lines.append(f'{k} & ' + ' & '.join(cells) + r' \\')
	lines += [r'\hline', r'\end{tabular}']
	if all_steps:
		(out / 'table_fid_steps.tex').write_text('\n'.join(lines) + '\n')

	# Macros for numbers quoted in the text.
	m = next(r for r in runs if r['name'] == main)
	s = main_value(m, main_steps)
	meta = m['fid_meta'] or {}
	fewest = min(m['steps'].values(), key=lambda r: r['num_steps'], default={})
	curve_n = next(iter(m['steps'].values()), {}).get('num_samples', meta.get('curve_samples'))
	gen = (m['classifier'] or {}).get('synthetic', {})
	real = (m['classifier'] or {}).get('real', {})
	best = min(m['steps'].values(), key=lambda r: r['fid'], default={})
	macros = {
		'EvalEpoch': str(m['epoch']),
		'NLLmain': fmt(m['nll'], 2) if m.get('nll_se') is None else f"${fmt(m['nll'], 2)}\\pm{fmt(m['nll_se'], 2)}$",
		'NLLcount': str(m['nll_count'] or '--'),
		'FIDmain': fmt(s.get('fid'), 1),
		'MainSteps': str(main_steps),
		'FIDbest': fmt(best.get('fid'), 1),
		'FIDbestSteps': str(best.get('num_steps', '--')),
		'FIDsamples': f"{s.get('num_samples', meta.get('generated_samples', 0)):,}".replace(',', r'\,'),
		'FIDcurveSamples': f"{curve_n or 0:,}".replace(',', r'\,'),
		'FIDkTen': fmt(m['steps'].get(10, {}).get('fid'), 1),
		'FIDrealMain': fmt(real_floor(m, curve_n), 2),
		'FIDfewest': fmt(fewest.get('fid'), 1),
		'FewestSteps': str(fewest.get('num_steps', '--')),
		'BalAccGen': fmt(gen.get('balanced_accuracy'), 3),
		'BalAccReal': fmt(real.get('balanced_accuracy'), 3),
		'AccGen': fmt(gen.get('accuracy'), 3),
		'AccReal': fmt(real.get('accuracy'), 3),
		'ConfDistance': fmt((m['classifier'] or {}).get('confusion_distance'), 3),
		'GuidanceScale': fmt(m['guidance_scale'], 1),
		'NNgen': fmt((m['nn'] or {}).get('generated_median'), 3),
		'NNheld': fmt((m['nn'] or {}).get('held_out_median'), 3),
		'NNfrac': fmt((m['nn'] or {}).get('fraction_generated_below_held_out_1st_percentile'), 3),
	}
	macros.update(ordinal_macros(m['ordinal']))
	for r in runs:
		key = macro_name(r['name'])
		macros[f'NLL{key}'] = fmt(r['nll'], 2)
		macros[f'FID{key}'] = fmt(main_value(r, main_steps).get('fid'), 1)
	text = '% Generated by conditional/collect_results.py; do not edit by hand.\n'
	text += '% Missing values are left out, so results/placeholders.tex shows them as ??.\n'
	text += ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in macros.items() if v != '--')
	(out / 'macros.tex').write_text(text)


def real_floor(run, num_a, num_b=50000):
	'''FID between num_a and num_b views of two disjoint halves of the real
	images (fid_real_reference.py), or None.'''
	ref = load(run['dir'] / 'metrics' / 'fid_real_reference.json')
	for r in (ref or {}).get('results', []):
		if (r.get('num_a', r.get('num_per_set')), r.get('num_b', r.get('num_per_set'))) == (num_a, num_b):
			return r['fid']
	return None


def ordinal_macros(o):
	'''Macros of ordinal_analysis.py (75-pattern and 11-group Hellinger 1-NN).'''
	if o is None:
		return {}
	p75, p11 = o.get('patterns75', {}), o.get('groups11', {})
	dist = o.get('distance_matrices', {}).get('patterns75', {})
	diag = dist.get('generated_diagonal_min_count')
	return {
		'OrdBalAccReal': fmt(p75.get('real', {}).get('balanced_accuracy'), 3),
		'OrdBalAccGen': fmt(p75.get('synthetic', {}).get('balanced_accuracy'), 3),
		'OrdAccReal': fmt(p75.get('real', {}).get('accuracy'), 3),
		'OrdAccGen': fmt(p75.get('synthetic', {}).get('accuracy'), 3),
		'OrdGroupBalAccReal': fmt(p11.get('real', {}).get('balanced_accuracy'), 3),
		'OrdGroupBalAccGen': fmt(p11.get('synthetic', {}).get('balanced_accuracy'), 3),
		'OrdDiagMin': '--' if diag is None else str(diag),
	}


def plot(runs, out):
	"""FID against DDIM steps (curve points), one line per model."""
	try:
		import matplotlib
		matplotlib.use('Agg')
		import matplotlib.pyplot as plt
		from matplotlib import font_manager
	except ImportError:
		print('matplotlib not available; skipping fid_vs_steps.pdf')
		return
	for f in font_manager.findSystemFonts():
		if 'texgyretermes' in f.lower():
			font_manager.fontManager.addfont(f)
	plt.rcParams.update({'font.family': 'serif',
		'font.serif': ['TeX Gyre Termes', 'Times New Roman', 'DejaVu Serif'],
		'font.size': 8, 'mathtext.fontset': 'stix', 'pdf.fonttype': 42})
	colors = {'x0': '#2a78d6', 'eps': '#eb6834'}
	label = {'x0': r'$x_0$ prediction', 'eps': r'$\epsilon$ prediction'}
	fig, ax = plt.subplots(figsize=(3.35, 2.3))
	ticks = set()
	shown = [r for r in runs if r['steps']]
	if not shown:
		print('no FID results; skipping fid_vs_steps.pdf')
		return
	for r in shown:
		k = sorted(r['steps'])
		ticks |= set(k)
		fid = [r['steps'][i]['fid'] for i in k]
		ax.plot(k, fid, '-o', color=colors.get(r['prediction_type'], '#1baf7a'), lw=2, ms=5,
			mec='white', mew=1, label=label.get(r['prediction_type'], r['name']))
		if len(shown) == 1:
			for x, y in zip(k, fid):
				ax.annotate(f'{y:.1f}', (x, y), textcoords='offset points', xytext=(0, 6),
					ha='center', fontsize=7)
	ax.set_ylabel('FID (lower is better)')
	ax.set_ylim(0, max(max(r['steps'][i]['fid'] for i in r['steps']) for r in shown) * 1.25)
	ax.set_xlabel('DDIM sampling steps $K$')
	ax.set_xscale('log')
	ax.set_xticks(sorted(ticks))
	ax.set_xticklabels([str(v) for v in sorted(ticks)])
	ax.minorticks_off()
	ax.grid(True, color='#e4e3df', lw=0.6)
	ax.set_axisbelow(True)
	for side in ('top', 'right'):
		ax.spines[side].set_visible(False)
	ax.tick_params(colors='#6b6a64', length=0)
	if len(shown) > 1:
		ax.legend(frameon=False, fontsize=7)
	fig.tight_layout(pad=0.4)
	fig.savefig(out / 'fid_vs_steps.pdf')
	plt.close(fig)


def main():
	p = argparse.ArgumentParser()
	p.add_argument('--runs', nargs='+', required=True)
	p.add_argument('--names', nargs='+', required=True)
	p.add_argument('--main', required=True, help='name of the model reported as main')
	p.add_argument('--main_steps', type=int, default=100)
	p.add_argument('--epoch', type=int, nargs='+', default=None,
		help='evaluated epoch (one value for all runs, or one per run)')
	p.add_argument('--classifier', nargs='+', default=['resnet50/logs/resnet50/test'],
		help='ResNet test folder: one for all runs, or one per run')
	p.add_argument('--out', default='../results')
	a = p.parse_args()
	if len(a.runs) != len(a.names):
		p.error('--runs and --names must have the same length')

	out = Path(a.out)
	out.mkdir(parents=True, exist_ok=True)
	epochs = a.epoch or [None]
	if len(epochs) == 1:
		epochs = epochs * len(a.runs)
	if len(epochs) != len(a.runs):
		p.error('--epoch takes one value or one per run')
	classifiers = a.classifier * len(a.runs) if len(a.classifier) == 1 else a.classifier
	if len(classifiers) != len(a.runs):
		p.error('--classifier takes one folder or one per run')
	runs = [collect(r, n, c, e) for r, n, c, e in zip(a.runs, a.names, classifiers, epochs)]
	for r, c in zip(runs, classifiers):
		r['classifier_dir'] = c
	write_tables(runs, a.main, a.main_steps, out)
	plot(runs, out)
	for r in runs:
		folder = Path(r['classifier_dir']) / r['dir'].name / f"epoch_{r['epoch']}"
		target = out / f"confusion_{r['name']}.pdf"
		drawn = False
		if (folder / 'confusion_matrices.npz').exists():
			# Article version (article_plots.py), panels side by side.
			try:
				from article_plots import plot_confusion
				plot_confusion(folder, str(target), layout='row')
				drawn = True
			except Exception as e:
				print(f'WARNING: could not draw {target.name}: {type(e).__name__}: {e}')
		if not drawn and (folder / 'confusion_matrices.pdf').exists():
			shutil.copy(folder / 'confusion_matrices.pdf', target)
		for name, target in (('nn_distances.pdf', 'nn_distances'), ('nearest_neighbours.pdf', 'nn_pairs'),
				('nn_summary.pdf', 'nn_summary')):
			src = r['metrics'] / name
			if src.exists():
				shutil.copy(src, out / f"{target}_{r['name']}.pdf")
		for target in ('ordinal_confusion', 'hc_plane', 'ordinal_distance'):
			src = r['metrics'] / 'ordinal' / f'{target}.pdf'
			if src.exists():
				shutil.copy(src, out / f"{target}_{r['name']}.pdf")
	print(f'LaTeX results written to {out.resolve()}')
	print((out / 'macros.tex').read_text())


if __name__ == '__main__':
	main()
