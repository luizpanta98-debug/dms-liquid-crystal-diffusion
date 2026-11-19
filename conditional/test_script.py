import os
import json
import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision.datasets.folder import find_classes
from unet import UNet
from dm import DM
from seed import set_seed
from data import get_checkpoint, apply_epoch_arg


def ddim_cond_sample(net, dm, config, batch_size=10, num_samples=100,
	out_dir='gen_samples', num_steps=None):
	'''
		num_steps: integer number of network evaluations, from 1 to dm.T
		num_samples: number of images per class
		config[eta]: 0 for deterministic DDIM; 1 for stochastic DDIM
		num_steps=None uses config['num_steps'] (100 in config.json)
	'''
	eta = config.get('eta', 1.0)
	if num_steps is None:
		num_steps = config.get('num_steps', 100)
	if not 1 <= num_steps <= dm.T:
		raise ValueError('num_steps must be between 1 and dm.T.')

	os.makedirs(out_dir, exist_ok=True)
	net.eval()

	subseq = np.linspace(dm.T-1, 0, num_steps, dtype=int).tolist()
	_, class_dict = find_classes(config['data_fp'])
	class_dict = {idx:classname for classname,idx in class_dict.items()}
	lbls = [idx for idx,_ in class_dict.items()]

	for lbl in lbls:
		img_num = 0
		with th.inference_mode():
			for start in range(0, num_samples, batch_size):
				B = min(batch_size, num_samples-start)
				xt = th.randn(
					B,
					config['in_channels'],
					config['img_resolution'],
					config['img_resolution'],
					device=config['device'])

				y = th.full((B,), lbl, dtype=th.long, device=config['device'])
				y = F.one_hot(y, num_classes=config['label_dim']).float()

				for idx,time in tqdm(list(enumerate(subseq))):
					ts = th.full((B,), time, dtype=th.long, device=config['device'])

					if idx < len(subseq)-1:
						tsprev = th.full(
							(B,), subseq[idx+1], dtype=th.long,
							device=config['device'])
						xt, x0, eps = dm.ddim_sample(
							eta, net, xt, ts, tsprev, y)
					else:
						# Finish at the clean image, rather than alpha_bar[0].
						x0, eps = dm.ddim_pred_x0(net, xt, ts, y)
						xt = x0.clamp(-1, 1)

				xt = xt.cpu().permute(0,2,3,1).numpy()
				for np_img in xt:
					np_img = (np_img + 1)/2
					np_img = (np_img * 255).clip(0, 255).astype(np.uint8)
					if config['in_channels'] == 1:
						np_img = np_img[:,:,0]
					img = Image.fromarray(np_img)
					img.save(f"{out_dir}/{class_dict[lbl]}_{img_num}.png")
					img_num+=1


if __name__ == "__main__":
	with open("config.json", "r") as f:
		config = json.load(f)
	apply_epoch_arg(config)

	set_seed(config['seed'])

	dm = DM(prediction_type=config.get('prediction_type', 'x0'), guidance_scale=config.get('guidance_scale', 1.5))
	net = UNet(
		img_resolution=config["img_resolution"],
		in_channels=config["in_channels"],
		out_channels=config["out_channels"],
		label_dim=config["label_dim"],
		model_channels=config["model_channels"],
		channel_mult=config["channel_mult"],
		channel_mult_emb=config["channel_mult_emb"],
		num_res_blocks=config["num_res_blocks"],
		attn_resolutions=config["attn_resolutions"],
		).to(config['device'])

	prefix = 'ema_weights' if config['ema'] else 'last_weights'
	model_path, epoch = get_checkpoint(
		f"{config['save_checkpoint']}/weights", prefix, config.get('eval_epoch'))
	net.load_state_dict(th.load(
		model_path, map_location=config['device'], weights_only=False))
	print(f'weights loaded: epoch {epoch}.')

	ddim_cond_sample(net, dm, config)