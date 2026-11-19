'''
Article figure: forward pass of the class-conditional U-Net (unet.py, config.json).

  noisy texture x_t  ->  encoder (6 levels)  ->  bottleneck  ->  decoder  ->  x0_hat
                    t, y -> embedding e, added to every residual block

Block height ~ spatial resolution, block width ~ number of channels.
Each thin slab is one residual block (2 per encoder level, 3 per decoder level).

Usage: python make_unet_figure.py <clean_texture.png> <out.pdf>
'''
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from PIL import Image

# Architecture (config.json): 64 x (1,1,2,2,4,4) channels, 2 res blocks, attention at 16x16.
RES = [256, 128, 64, 32, 16, 8]
CH = [64, 64, 128, 128, 256, 256]
ATTN = {16}

ENC, DEC, ATT = '#2a78d6', '#1baf7a', '#eb6834'   # validated categorical slots
INK, MUTED, LINE = '#1f1f1e', '#6b6a64', '#b9b8b2'

from matplotlib import font_manager
for _f in font_manager.findSystemFonts():
	if 'texgyretermes' in _f.lower():
		font_manager.fontManager.addfont(_f)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['TeX Gyre Termes', 'Times New Roman', 'DejaVu Serif'],
	'font.size': 8, 'mathtext.fontset': 'stix',
	'pdf.fonttype': 42})


def tint(hex_color, a):
	c = np.array(matplotlib.colors.to_rgb(hex_color))
	return tuple(1 - a * (1 - c))


def slab(ax, x, yc, w, h, color, attn=False):
	ax.add_patch(FancyBboxPatch((x, yc - h / 2), w, h, boxstyle='round,pad=0,rounding_size=0.03',
		fc=tint(color, 0.55), ec=color, lw=0.8, zorder=3))
	# embedding injection: small dot on top of every residual block
	ax.add_patch(Circle((x + w / 2, yc + h / 2 + 0.07), 0.035, fc=MUTED, ec='none', zorder=4))
	if attn:
		ax.add_patch(FancyBboxPatch((x - 0.025, yc - h / 2 - 0.025), w + 0.05, h + 0.05,
			boxstyle='round,pad=0,rounding_size=0.04', fc='none', ec=ATT, lw=1.4, zorder=5))


def arrow(ax, p, q, color=MUTED, lw=1.0, ls='-', head=6, rad=0.0, z=2):
	ax.add_patch(FancyArrowPatch(p, q, arrowstyle=f'-|>,head_length={head/20},head_width={head/40}',
		mutation_scale=10, color=color, lw=lw, ls=ls, zorder=z,
		connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0))


def main(texture_path, out_path):
	rng = np.random.default_rng(3)
	x0 = np.asarray(Image.open(texture_path).convert('RGB').resize((256, 256)), dtype=float) / 127.5 - 1
	# forward process at t = 300 of the linear schedule (beta 1e-4 .. 2e-2, T = 1000)
	beta = np.linspace(1e-4, 2e-2, 1000)
	abar = np.cumprod(1 - beta)[299]
	xt = np.sqrt(abar) * x0 + np.sqrt(1 - abar) * rng.standard_normal(x0.shape)
	show = lambda a: np.clip((a + 1) / 2, 0, 1)

	fig = plt.figure(figsize=(7.1, 3.55))
	ax = fig.add_axes([0, 0, 1, 1])
	ax.set_xlim(0, 14.2)
	ax.set_ylim(-0.85, 6.25)
	ax.set_aspect('equal')
	ax.axis('off')

	# ---- input and output images ----
	img_w = 1.75
	iy = 4.55 - img_w / 2
	ax.imshow(show(xt), extent=(0.15, 0.15 + img_w, iy, iy + img_w), zorder=3)
	ax.imshow(show(x0), extent=(14.2 - 0.15 - img_w, 14.2 - 0.15, iy, iy + img_w), zorder=3)
	ax.text(0.15 + img_w / 2, iy + img_w + 0.12, r'noisy texture $x_t$', ha='center', va='bottom', color=INK)
	ax.text(14.05 - img_w / 2, iy + img_w + 0.12, r'prediction $\hat{x}_{0}$', ha='center', va='bottom', color=INK)
	ax.text(0.15 + img_w / 2, iy - 0.12, r'$256\times256\times3$', ha='center', va='top', color=MUTED, fontsize=7)
	ax.text(14.05 - img_w / 2, iy - 0.12, r'$256\times256\times3$', ha='center', va='top', color=MUTED, fontsize=7)

	# ---- geometry of the U ----
	top = 4.55                        # centre of the 256x256 level
	dy = 0.76                         # vertical step between levels
	heights = [1.10, 0.92, 0.76, 0.62, 0.50, 0.40]
	widths = [0.11 + 0.07 * np.log2(c / 64) for c in CH]
	gap = 0.06
	cxm = 7.1                         # horizontal centre of the U

	def offset(lvl):                  # horizontal distance of a level from the centre
		return 0.75 + (len(RES) - 1 - lvl) * 0.78

	enc, dec = [], []
	for lvl, (r, h, w) in enumerate(zip(RES, heights, widths)):
		y = top - lvl * dy
		bw = 2 * w + gap
		x = cxm - offset(lvl) - bw / 2
		for k in range(2):
			slab(ax, x + k * (w + gap), y, w, h, ENC, attn=(r in ATTN))
		enc.append((x, x + bw, y, h))
		ax.text(x + bw / 2, y - h / 2 - 0.07, rf'${r}^2$', ha='center', va='top', fontsize=6.5, color=MUTED)
		bw = 3 * w + 2 * gap
		x = cxm + offset(lvl) - bw / 2
		for k in range(3):
			slab(ax, x + k * (w + gap), y, w, h, DEC, attn=(r in ATTN and k == 2))
		dec.append((x, x + bw, y, h))
		ax.text(x + bw / 2, y - h / 2 - 0.07, rf'${r}^2$', ha='center', va='top', fontsize=6.5, color=MUTED)
		# skip connection at the same resolution (concatenation)
		ax.add_patch(FancyArrowPatch((enc[-1][1] + 0.05, y + 0.1), (x - 0.05, y + 0.1),
			arrowstyle='-|>,head_length=0.25,head_width=0.12', mutation_scale=10, color=LINE, lw=0.9,
			ls=(0, (3, 2)), zorder=1, shrinkA=0, shrinkB=0))

	# downsampling (encoder) and upsampling (decoder) between levels
	for lvl in range(len(RES) - 1):
		x0e, x1e, ye, he = enc[lvl]
		nx0, _, ny, _ = enc[lvl + 1]
		arrow(ax, (x1e - 0.02, ye - he / 2 - 0.02), (nx0 - 0.04, ny), color=ENC, lw=1.0)
		x0d, x1d, yd, hd = dec[lvl]
		_, px1, py, _ = dec[lvl + 1]
		arrow(ax, (px1 + 0.04, py), (x0d + 0.02, yd - hd / 2 - 0.02), color=DEC, lw=1.0)

	# bottleneck: residual block with attention, then residual block
	yb = top - len(RES) * dy
	wb, hb = widths[-1], heights[-1]
	xb = cxm - (2 * wb + gap) / 2
	slab(ax, xb, yb, wb, hb, MUTED, attn=True)
	slab(ax, xb + wb + gap, yb, wb, hb, MUTED)
	ax.text(cxm, yb - hb / 2 - 0.07, r'$8^2$', ha='center', va='top', fontsize=6.5, color=MUTED)
	x0e, x1e, ye, he = enc[-1]
	arrow(ax, (x1e - 0.02, ye - he / 2 - 0.02), (xb - 0.05, yb), color=MUTED, lw=1.0)
	x0d, x1d, yd, hd = dec[-1]
	arrow(ax, (xb + 2 * wb + gap + 0.05, yb), (x0d + 0.02, yd - hd / 2 - 0.02), color=MUTED, lw=1.0)

	# image -> first block, last block -> image
	arrow(ax, (0.15 + img_w + 0.06, top), (enc[0][0] - 0.06, top), color=INK, lw=1.1)
	arrow(ax, (dec[0][1] + 0.06, top), (14.2 - 0.15 - img_w - 0.06, top), color=INK, lw=1.1)

	# channel counts, written once per distinct width
	for lvl in (0, 2, 4):
		x0e, x1e, ye, he = enc[lvl]
		ax.text((x0e + x1e) / 2, ye + he / 2 + 0.17, f'{CH[lvl]} ch', ha='center', va='bottom', fontsize=6.5, color=MUTED)

	# ---- conditioning branch ----
	cx, cy = 6.4, 6.0
	ax.text(cx - 1.55, cy, r'timestep $t$', ha='right', va='center', color=INK)
	ax.text(cx - 1.55, cy - 0.42, r'phase label $y$', ha='right', va='center', color=INK)
	box = dict(boxstyle='round,pad=0.25', fc='white', ec=LINE, lw=0.8)
	ax.text(cx - 0.95, cy, 'sinusoidal', ha='center', va='center', fontsize=6.5, color=INK, bbox=box)
	ax.text(cx - 0.95, cy - 0.42, 'linear', ha='center', va='center', fontsize=6.5, color=INK, bbox=box)
	ax.text(cx - 0.12, cy - 0.21, r'$+$', ha='center', va='center', fontsize=9, color=INK)
	ax.text(cx + 0.55, cy - 0.21, 'MLP', ha='center', va='center', fontsize=6.5, color=INK, bbox=box)
	arrow(ax, (cx - 0.5, cy), (cx - 0.22, cy - 0.14), color=MUTED, lw=0.8, head=4)
	arrow(ax, (cx - 0.5, cy - 0.42), (cx - 0.22, cy - 0.28), color=MUTED, lw=0.8, head=4)
	arrow(ax, (cx - 0.02, cy - 0.21), (cx + 0.3, cy - 0.21), color=MUTED, lw=0.8, head=4)
	ax.add_patch(Circle((cx + 1.22, cy - 0.21), 0.05, fc=MUTED, ec='none'))
	arrow(ax, (cx + 0.82, cy - 0.21), (cx + 1.15, cy - 0.21), color=MUTED, lw=0.8, head=4)
	ax.text(cx + 1.33, cy - 0.21, r'$e$ (added to every block, $\bullet$)', ha='left', va='center',
		fontsize=7, color=INK)

	# ---- legend ----
	ly = -0.68
	def key(x, color, label, fill=True, lw=0.8):
		ax.add_patch(FancyBboxPatch((x, ly - 0.11), 0.1, 0.22, boxstyle='round,pad=0,rounding_size=0.02',
			fc=tint(color, 0.55) if fill else 'white', ec=color, lw=lw))
		ax.text(x + 0.2, ly, label, va='center', fontsize=7, color=INK)
	key(1.3, ENC, 'encoder residual block')
	key(4.2, DEC, 'decoder residual block')
	key(7.1, MUTED, 'bottleneck')
	key(8.65, ATT, 'with self-attention', fill=False, lw=1.4)
	ax.add_patch(FancyArrowPatch((11.0, ly), (11.45, ly), arrowstyle='-|>,head_length=0.25,head_width=0.12',
		mutation_scale=10, color=LINE, lw=0.9, ls=(0, (3, 2))))
	ax.text(11.55, ly, 'skip (concatenation)', va='center', fontsize=7, color=INK)

	fig.savefig(out_path)
	fig.savefig(out_path.rsplit('.', 1)[0] + '.png', dpi=250)
	print(f'saved -> {out_path}')


if __name__ == '__main__':
	main(sys.argv[1], sys.argv[2])
