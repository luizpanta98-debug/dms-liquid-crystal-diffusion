from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset, random_split
from PIL import Image
from copy import copy
import json
import os
from pathlib import Path


class FourCropImageFolder(ImageFolder):
    def __len__(self):
        return 4 * super().__len__()

    def __getitem__(self, index):
        image, target = super().__getitem__(index // 4)
        size = self.crop_size
        left = (index % 2) * (image.width - size)
        top = (index % 4 // 2) * (image.height - size)
        return to_tensor(image.crop((left, top, left + size, top + size))), target


def get_checkpoint_epoch(path):
	suffix = Path(path).stem.rsplit('_', 1)[-1]
	return int(suffix) if suffix.isdigit() else None


def get_latest_checkpoint(weights_dir, prefix='last_weights'):
	candidates = []
	for path in Path(weights_dir).glob(f'{prefix}_*.pt'):
		suffix = path.stem[len(prefix) + 1:]
		if path.is_file() and suffix.isdigit():
			candidates.append((get_checkpoint_epoch(path), path))

	if not candidates:
		raise FileNotFoundError(
			f'No numbered {prefix} checkpoints found in {weights_dir}. '
			'A legacy latest checkpoint does not identify its epoch.')

	epoch, path = max(candidates, key=lambda item: item[0])
	return str(path), epoch


def get_checkpoint(weights_dir, prefix, epoch=None):
	"""Checkpoint of a given epoch, or the latest one when epoch is None.

	The epoch comes from "eval_epoch" in config.json. Later epochs are not
	always better (e.g. 500 epochs gave worse samples than 100), so the
	evaluated epoch is chosen explicitly and recorded with every result.
	"""
	if epoch is None:
		return get_latest_checkpoint(weights_dir, prefix)
	if type(epoch) is not int or epoch < 0:
		raise ValueError('eval_epoch must be a non-negative integer or null.')
	path = Path(weights_dir) / f'{prefix}_{epoch}.pt'
	if not path.is_file():
		raise FileNotFoundError(f'Checkpoint not found: {path}')
	return str(path), epoch


def apply_epoch_arg(config, argv=None):
	"""Set config['eval_epoch'] from '--epoch N' on the command line, if given.

	Prints which epoch will be evaluated, and warns when neither the command
	line nor config.json specifies one (the latest checkpoint is then used).
	"""
	import argparse
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument('--epoch', type=int, default=None)
	args, _ = parser.parse_known_args(argv)
	if args.epoch is not None:
		config['eval_epoch'] = args.epoch
	if config.get('eval_epoch') is None:
		print('WARNING: no "eval_epoch" in config.json and no --epoch given; '
			'the LATEST checkpoint will be evaluated.')
	else:
		print(f"Evaluating checkpoint of epoch {config['eval_epoch']}.")
	return config


def metrics_dir(config, epoch):
	"""Results of each evaluated epoch go to their own folder."""
	path = os.path.join(config['save_checkpoint'], 'metrics', f'epoch_{epoch}')
	os.makedirs(path, exist_ok=True)
	return path


class ImageDataset(Dataset):
	def __init__(self, folder, transform=None):
		self.folder = folder
		self.transform = transform

		self.files = [
			os.path.join(folder, f)
			for f in os.listdir(folder)
			if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
		]

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		image = Image.open(self.files[idx]).convert("RGB")
		if self.transform:
			image = self.transform(image)
		return image


# util for weighted sampler
def get_subset_targets(subset):
	return [subset.dataset.targets[i] for i in subset.indices]


def is_valid_file(path):
	try:
		with Image.open(path) as img:
			img.verify()
		return True
	except Exception:
		print(f"Skipping damaged file: {path}")
		return False


def load_imagefolder(fp, img_resolution, split, val_fp="val_files.json"):
	train_transform = transforms.Compose([
		transforms.RandomHorizontalFlip(0.5),
		transforms.RandomVerticalFlip(0.5),
		transforms.RandomCrop(img_resolution),
		transforms.ToTensor(), #[0,1]
		transforms.Normalize(mean=[0.5]*3, std=[0.5]*3) #[-1,1]
		])

	val_transform = transforms.Compose([
		transforms.CenterCrop(img_resolution),
		transforms.ToTensor(), #[0,1]
		transforms.Normalize(mean=[0.5]*3, std=[0.5]*3) #[-1,1]
		])

	dataset = ImageFolder(
		root=fp, transform=train_transform, is_valid_file=is_valid_file)
	train_size = int(split * len(dataset))
	val_size = len(dataset) - train_size

	train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

	val_files = [os.path.abspath(dataset.samples[i][0]) for i in val_dataset.indices]
	with open(val_fp, "w") as f:
		json.dump(val_files, f, indent=4)

	# Separate transforms, preserving file ordering and labels.
	val_dataset.dataset = copy(dataset)
	val_dataset.dataset.transform = val_transform

	return train_dataset, val_dataset


def load_imagedataset(fp, img_resolution, split):
	transform = transforms.Compose([
		transforms.Resize((img_resolution, img_resolution)),
		transforms.ToTensor(), #[0,1]
		transforms.Normalize(mean=[0.5]*3, std=[0.5]*3) #[-1,1]
		])

	dataset = ImageDataset(fp, transform=transform)
	train_size = int(split * len(dataset))
	val_size = len(dataset) - train_size
	train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

	return train_dataset, val_dataset