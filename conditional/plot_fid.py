'''
FID of the model at the global sampling setting (config.json: num_steps=100).

Protocol (defaults, change on the command line):
  * curve: K in NUM_STEPS_VALUES (10, 25, 50, 100), --curve_samples 10000
    generated images per K
  * main value: --main_samples 10000 images at --main_steps K (default: num_steps
    of config.json), i.e. the curve point at K = 100; FID and KID are reported
  * reference: 50,000 real crop/flip views (features computed once)
  * floor: fid_real_reference.py computes FID and KID between two disjoint halves
    of the real images with the same sizes (10,000 vs 50,000 views)
Every FID is therefore FID(10k generated, 50k real): one N for all numbers.
KID (Binkowski et al. 2018) is unbiased in N; 100 subsets of 1000 images.

Results are written after every K to <save_checkpoint>/metrics/epoch_<N>/
fid_and_inception.json, and K values already in that file are skipped, so an
interrupted run can simply be restarted.

Speed: bf16 autocast for the U-Net (--precision fp32 to disable), batch of
--batch images (guidance doubles it inside dm.py), TF32 and cuDNN autotuning.
Inception features are always computed in fp32.

Example:  python plot_fid.py --epoch 100
          python plot_fid.py --epoch 100 --steps        # main value only, no curve
'''
# pip install pytorch-fid matplotlib scienceplots

import os
# Less fragmentation when the GPU is shared; set before torch is imported.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json
import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from pytorch_fid.inception import fid_inception_v3
from pytorch_fid.fid_score import calculate_frechet_distance
from importlib.metadata import version
from tqdm import tqdm
import matplotlib.pyplot as plt

from unet import UNet
from dm import DM
from seed import set_seed
from data import is_valid_file, get_checkpoint, metrics_dir, apply_epoch_arg


NUM_STEPS_VALUES = [10, 25, 50, 100]   # FID-vs-steps curve; 100 = num_steps is also the main value
NUM_SAMPLES = 50000          # real reference views
GEN_SAMPLES = 10000          # generated images per K (curve and main value)
KID_SUBSETS, KID_SUBSET_SIZE = 100, 1000
GEN_BATCH = 4                # overridden by --batch (guidance doubles it: 8 on the GPU)
FEAT_BATCH = 50


class CropDataset(Dataset):
	def __init__(self, dataset, indices, img_resolution):
		self.dataset = dataset
		self.indices = indices
		self.img_resolution = img_resolution
		self.transform = transforms.PILToTensor()

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, idx):
		source_idx, view = divmod(int(self.indices[idx]), 16)
		corner, flip = divmod(view, 4)
		image, label = self.dataset[source_idx]

		size = self.img_resolution
		offset = 600-size
		positions = [(0,0), (0,offset), (offset,0), (offset,offset)]
		top, left = positions[corner]
		image = image.crop((left, top, left+size, top+size))

		if flip & 1:
			image = transforms.functional.hflip(image)
		if flip & 2:
			image = transforms.functional.vflip(image)

		return self.transform(image), label


def load_config():
	with open('config.json', 'r') as f:
		return apply_epoch_arg(json.load(f))


def build_net(config):
	net = UNet(
		img_resolution=config['img_resolution'],
		in_channels=config['in_channels'],
		out_channels=config['out_channels'],
		label_dim=config['label_dim'],
		model_channels=config['model_channels'],
		channel_mult=config['channel_mult'],
		channel_mult_emb=config['channel_mult_emb'],
		num_res_blocks=config['num_res_blocks'],
		attn_resolutions=config['attn_resolutions'],
	).to(config['device'])

	prefix = 'ema_weights' if config['ema'] else 'last_weights'
	model_path, epoch = get_checkpoint(
		f"{config['save_checkpoint']}/weights", prefix, config.get('eval_epoch'))
	net.load_state_dict(th.load(
		model_path, map_location=config['device'], weights_only=False))
	print(f'weights loaded: epoch {epoch}.')
	net.eval()
	return net, model_path, epoch


def real_dataset_and_counts(config):
	dataset = ImageFolder(root=config['data_fp'], is_valid_file=is_valid_file)
	counts = np.bincount(dataset.targets, minlength=config['label_dim'])
	return dataset, counts


def sample_labels(counts, seed, num=NUM_SAMPLES):
	# Exactly num labels, preserving the real class proportions, in random order
	# (so any prefix is itself a random, nearly proportional subset).
	quota = num * counts / counts.sum()
	gen_counts = np.floor(quota).astype(int)
	order = np.argsort(-(quota-gen_counts), kind='stable')
	gen_counts[order[:num-gen_counts.sum()]] += 1

	labels = np.repeat(np.arange(len(counts)), gen_counts)
	rng = np.random.RandomState(seed)
	rng.shuffle(labels)
	return labels, gen_counts


def crop_reference(dataset, counts, config):
	# Each source/corner/flip combination is selected at most once.
	rng = np.random.RandomState(config['seed'])
	targets = np.array(dataset.targets)
	indices = []

	for label,n in enumerate(counts):
		sources = np.flatnonzero(targets == label)
		views = (sources[:,None]*16 + np.arange(16)).ravel()
		indices.append(rng.choice(views, size=int(n), replace=False))

	indices = np.concatenate(indices)
	rng.shuffle(indices)
	return CropDataset(dataset, indices, config['img_resolution'])


def get_inception_model(device):
	# pytorch-fid Inception up to the 2048-d pool features (no classifier head).
	model = fid_inception_v3()
	model.fc = th.nn.Identity()
	model.transform_input = False
	return model.to(device).eval()


@th.inference_mode()
def inception_features(imgs, model):
	# uint8 RGB, NCHW; preprocessing follows pytorch-fid.
	x = imgs.float()/255
	x = F.interpolate(x, size=(299,299), mode='bilinear', align_corners=False)
	return model(2*x-1).cpu().numpy()


@th.inference_mode()
def real_features(dataset, device, model):
	loader = DataLoader(
		dataset, batch_size=FEAT_BATCH, shuffle=False,
		num_workers=min(8, os.cpu_count() or 1))

	feats = []
	for imgs, _ in tqdm(loader, desc='real features'):
		feats.append(inception_features(imgs.to(device), model))

	return np.concatenate(feats, axis=0)


def stats_of(feats):
	return feats.mean(axis=0, dtype=np.float64), np.cov(feats, rowvar=False)


def real_stats(dataset, device, model):
	return stats_of(real_features(dataset, device, model))


def kid(feats_a, feats_b, seed=0, subsets=KID_SUBSETS, size=KID_SUBSET_SIZE):
	'''Kernel Inception Distance: unbiased MMD^2 with the cubic polynomial kernel
	k(x,y) = (x.y/d + 1)^3, averaged over random subsets (as in StyleGAN2-ADA).
	Returns (mean, standard deviation over subsets).'''
	rng = np.random.RandomState(seed)
	d = feats_a.shape[1]
	m = min(len(feats_a), len(feats_b), size)
	values = []
	for _ in range(subsets):
		x = feats_a[rng.choice(len(feats_a), m, replace=False)].astype(np.float64)
		y = feats_b[rng.choice(len(feats_b), m, replace=False)].astype(np.float64)
		kxx = (x @ x.T / d + 1) ** 3
		kyy = (y @ y.T / d + 1) ** 3
		kxy = (x @ y.T / d + 1) ** 3
		values.append((kxx.sum() - np.trace(kxx) + kyy.sum() - np.trace(kyy)) / (m*(m-1))
			- 2*kxy.mean())
	return float(np.mean(values)), float(np.std(values))


@th.inference_mode()
def sample_features(net, dm, config, num_steps, labels, model,
		batch=GEN_BATCH, precision='bf16'):
	dev = config['device']
	use_amp = precision == 'bf16' and th.device(dev).type == 'cuda'
	eta = config.get('eta', 1.0)
	subseq = np.linspace(dm.T-1, 0, num_steps, dtype=int).tolist()

	# Restart initial noise independently of intermediate sampling noise.
	gen = th.Generator(device=dev).manual_seed(config['seed'])
	feats = []

	for start in tqdm(range(0, len(labels), batch), desc=f'{num_steps} steps'):
		label_batch = labels[start:start+batch]
		B = len(label_batch)
		y = th.tensor(label_batch, dtype=th.long, device=dev)
		y = F.one_hot(y, num_classes=config['label_dim']).float()

		xt = th.randn(
			B,
			config['in_channels'],
			config['img_resolution'],
			config['img_resolution'],
			device=dev, generator=gen)

		# The U-Net runs in bf16; dm.py casts its output back to fp32, so the
		# DDIM update itself stays in fp32.
		with th.autocast('cuda', dtype=th.bfloat16, enabled=use_amp):
			for idx,time in enumerate(subseq):
				ts = th.full((B,), time, dtype=th.long, device=dev)
				if idx < len(subseq)-1:
					tsprev = th.full(
						(B,), subseq[idx+1], dtype=th.long, device=dev)
					xt, x0, eps = dm.ddim_sample(
						eta, net, xt, ts, tsprev, y)
				else:
					x0, eps = dm.ddim_pred_x0(net, xt, ts, y)
					xt = x0.clamp(-1, 1)
		xt = xt.float()

		# Same uint8 quantization as test_script.py.
		imgs = ((xt+1)*127.5).clamp(0, 255).to(th.uint8)
		feats.append(inception_features(imgs, model))

	return np.concatenate(feats, axis=0)


def set_style():
	try:
		import scienceplots
		plt.style.use(['science', 'no-latex'])
	except ImportError:
		plt.rcParams.update({
			'font.family': 'serif',
			'font.size': 11,
			'axes.grid': True,
			'grid.alpha': 0.3,
			'axes.spines.top': False,
			'axes.spines.right': False,
		})
	plt.rcParams['figure.dpi'] = 150


def real_floor(config, num_a, num_b=NUM_SAMPLES):
	'''FID between num_a views of half A and num_b views of half B of the real
	images (fid_real_reference.py), or None if not computed.'''
	path = os.path.join(config['save_checkpoint'], 'metrics', 'fid_real_reference.json')
	if not os.path.isfile(path):
		return None
	with open(path) as f:
		results = json.load(f)['results']
	for r in results:
		a, b = r.get('num_a', r.get('num_per_set')), r.get('num_b', r.get('num_per_set'))
		if (a, b) == (num_a, num_b):
			return r['fid']
	return None


def plot_fid(num_steps_values, fid_values, out_path, floor=None):
	fig, ax = plt.subplots(figsize=(4.5, 3.5))
	ax.plot(num_steps_values, fid_values, marker='o', linewidth=1.8, label='Generated vs real')
	for k, v in zip(num_steps_values, fid_values):
		ax.annotate(f'{v:.1f}', (k, v), textcoords='offset points', xytext=(0, 6), ha='center')
	if floor is not None:
		ax.axhline(floor, linestyle='--', color='0.4', linewidth=1,
			label=f'Real vs real ({floor:.1f})')
		ax.legend(frameon=False)
	ax.set_xscale('log')
	ax.set_xlabel('DDIM sampling steps')
	ax.set_ylabel('FID')
	ax.set_ylim(bottom=0)
	ax.set_xticks(num_steps_values)
	ax.set_xticklabels(num_steps_values)
	ax.minorticks_off()
	fig.tight_layout()
	fig.savefig(out_path)
	fig.savefig(os.path.splitext(out_path)[0]+'.pdf')
	plt.close(fig)


def parse_args():
	import argparse
	p = argparse.ArgumentParser(description='FID at num_steps (optional FID-vs-steps curve).')
	p.add_argument('--epoch', type=int, default=None, help='checkpoint epoch (overrides eval_epoch)')
	p.add_argument('--steps', type=int, nargs='*', default=NUM_STEPS_VALUES,
		help='step counts for the FID-vs-steps curve (default 10 25 50 100; '
			'pass --steps with no values for the main result only)')
	p.add_argument('--curve_samples', type=int, default=GEN_SAMPLES)
	p.add_argument('--main_steps', type=int, default=None,
		help='K of the main result (default: num_steps in config.json)')
	p.add_argument('--main_samples', type=int, default=GEN_SAMPLES)
	p.add_argument('--batch', type=int, default=GEN_BATCH)
	p.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
	return p.parse_args()


def fid_of(feats, mu_real, sigma_real):
	mu = feats.mean(axis=0, dtype=np.float64)
	sigma = np.cov(feats, rowvar=False)
	return float(calculate_frechet_distance(mu_real, sigma_real, mu, sigma))


def main():
	args = parse_args()
	config = load_config()
	device = config['device']
	set_seed(config['seed'])
	# Speed over bit-wise reproducibility: seeds still fix the noise, but cuDNN
	# may choose different (faster) kernels between runs.
	th.use_deterministic_algorithms(False)
	th.backends.cudnn.deterministic = False
	th.backends.cudnn.benchmark = True
	th.backends.cuda.matmul.allow_tf32 = True
	th.backends.cudnn.allow_tf32 = True

	if args.main_steps is None:
		args.main_steps = config.get('num_steps', 100)
	steps = sorted(set(args.steps) | {args.main_steps})
	if any(not 1 <= n <= 1000 for n in steps):
		raise ValueError('Sampling step counts must be between 1 and T.')
	if args.curve_samples > args.main_samples:
		raise ValueError('--curve_samples cannot exceed --main_samples.')

	dm = DM(prediction_type=config.get('prediction_type', 'x0'), guidance_scale=config.get('guidance_scale', 1.5))
	net, model_path, epoch = build_net(config)
	# One folder per model and epoch: runs and epochs never overwrite each other.
	out_dir = metrics_dir(config, epoch)
	out_json = os.path.join(out_dir, 'fid_and_inception.json')

	protocol = {
		'curve_samples': args.curve_samples, 'main_steps': args.main_steps,
		'main_samples': args.main_samples, 'precision': args.precision,
		'guidance_scale': dm.guidance_scale, 'eta': config.get('eta', 1.0),
		'checkpoint': model_path,
	}
	# Curve points can be reused if everything that affects them is unchanged
	# (main_samples only affects the main value, which is recomputed if needed).
	curve_keys = ('curve_samples', 'precision', 'guidance_scale', 'eta', 'checkpoint')
	report = None
	if os.path.exists(out_json):
		old = json.load(open(out_json))
		old_protocol = old.get('protocol', {})
		if all(old_protocol.get(k) == protocol[k] for k in curve_keys):
			report = old
			if old_protocol != protocol:
				report['protocol'] = protocol
				report['generated_samples'] = args.main_samples
				report['main'] = None
				report['results'] = [r for r in report['results']
					if r['num_steps'] != args.main_steps]
			print(f"resuming: already done K = {[r['num_steps'] for r in report['results']]}")
		else:
			backup = out_json.replace('.json', '_previous.json')
			os.replace(out_json, backup)
			print(f'different protocol in existing results; moved to {backup}')

	dataset, counts = real_dataset_and_counts(config)
	# One label sequence of length max(main, 50000); every K uses a prefix of it,
	# so the curve and the main value share labels (and initial noise) image by image.
	labels_all = sample_labels(counts, config['seed'], max(args.main_samples, NUM_SAMPLES))[0]
	labels_main = labels_all[:args.main_samples]
	labels_curve = labels_all[:args.curve_samples]
	reference = crop_reference(dataset, sample_labels(counts, config['seed'], NUM_SAMPLES)[1], config)

	print(f'source images: {len(dataset)}, class counts={counts.tolist()}')
	print(f'real patches: {len(reference)}; curve: {args.curve_samples} images per K; '
		f'main: {args.main_samples} images at K={args.main_steps}')

	model = get_inception_model(device)
	feats_real = real_features(reference, device, model)
	mu_real, sigma_real = stats_of(feats_real)
	print('real features computed.')

	if report is None:
		report = {
			'protocol': protocol,
			'config': config,
			'data_fp': config['data_fp'],
			'checkpoint': model_path,
			'epoch': epoch,
			'sampler': 'DDIM',
			'eta': config.get('eta', 1.0),
			'T': dm.T,
			'beta_1': dm.beta_1,
			'beta_T': dm.beta_T,
			'reference': 'four corner crops with flip orientations, sampled without replacement',
			'real_source_images': len(dataset),
			'real_samples': len(reference),
			'generated_samples': args.main_samples,
			'curve_samples': args.curve_samples,
			'class_to_idx': dataset.class_to_idx,
			'real_source_class_counts': counts.tolist(),
			'seed': config['seed'],
			'gen_batch': args.batch,
			'quantization': 'uint8, clip and truncate',
			'feature_extractor': 'pytorch-fid Inception',
			'pytorch_fid_version': version('pytorch-fid'),
			'torch_version': th.__version__,
			'torchvision_version': version('torchvision'),
			'numpy_version': np.__version__,
			'scipy_version': version('scipy'),
			'results': [],
			'main': None,
		}

	done = {r['num_steps'] for r in report['results']}
	for num_steps in steps:
		if num_steps in done:
			continue
		is_main = num_steps == args.main_steps
		labels = labels_main if is_main else labels_curve
		# Restart the random sequence for each sampling configuration.
		set_seed(config['seed'])
		th.use_deterministic_algorithms(False)
		th.backends.cudnn.deterministic = False
		th.backends.cudnn.benchmark = True
		feats = sample_features(net, dm, config, num_steps, labels, model,
			batch=args.batch, precision=args.precision)

		# Curve point: always the first curve_samples images (same N for every K).
		n = args.curve_samples
		fid = fid_of(feats[:n], mu_real, sigma_real)
		print(f'num_steps={num_steps}: FID={fid:.3f}  (N={n})')
		report['results'].append({'num_steps': num_steps, 'num_samples': n, 'fid': fid})
		if is_main:
			fid = fid_of(feats, mu_real, sigma_real)
			kid_mean, kid_std = kid(feats, feats_real, seed=config['seed'])
			print(f'main K={num_steps}: FID={fid:.3f}  KID={kid_mean:.5f} +- {kid_std:.5f}'
				f'  (N={len(feats)})')
			report['main'] = {'num_steps': num_steps, 'num_samples': len(feats), 'fid': fid,
				'kid': kid_mean, 'kid_std_over_subsets': kid_std,
				'kid_subsets': KID_SUBSETS, 'kid_subset_size': KID_SUBSET_SIZE}
			np.save(os.path.join(out_dir, f'inception_features_K{num_steps}.npy'), feats)
		del feats

		# Save after every K: an interrupted run keeps everything finished so far.
		report['results'].sort(key=lambda r: r['num_steps'])
		with open(out_json, 'w') as f:
			json.dump(report, f, indent=4)
		set_style()
		ks = [r['num_steps'] for r in report['results']]
		# The floor uses the same sizes as the curve: curve_samples vs NUM_SAMPLES.
		plot_fid(ks, [r['fid'] for r in report['results']],
			os.path.join(out_dir, 'fid_vs_num_steps.png'),
			floor=real_floor(config, args.curve_samples, NUM_SAMPLES))
		print(f'saved -> {out_dir}', flush=True)


if __name__ == '__main__':
	main()