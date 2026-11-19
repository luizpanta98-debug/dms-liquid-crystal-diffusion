# Conditional diffusion model for liquid-crystal textures (256 × 256)

A small implementation of a class-conditional DDPM/DDIM; Section S4 of the
ESI of the article maps each equation to the code. Each file does one thing:

| File | Content |
|---|---|
| `dm.py` | Linear schedule, forward process `q`, ancestral step `p`, DDIM step, classifier-free guidance, variational NLL bound |
| `unet.py` | Reduced EDM U-Net (≈42.5 M parameters with `config.json`) |
| `data.py` | Image folder loading, random 256 × 256 crops and flips, validation split |
| `train_script.py` | Training with Min-SNR-5 weighting, class-balanced sampling, label dropout, EMA |
| `test_script.py` | Draws sample grids with DDIM |
| `nll_calc.py` | NLL bound (bits/dim) on the corner crops of every validation image (`--crops 4` or `1`) |
| `plot_fid.py` | FID of 10,000 generated images against 50,000 real crops at K = 10, 25, 50, 100 (`--steps`); figure of FID against K with the real-vs-real floor; saves after every K and resumes |
| `fid_real_reference.py` | FID between two disjoint halves of the real images (10,000 vs 50,000 views): the value of an ideal generator under the same protocol |
| `resnet50/` | ResNet-50 phase classifier, confusion matrices on real and generated images |
| `nearest_neighbours.py` | Memorisation check: distance of generated images to the closest training crop |
| `ordinal.py`, `ordinal_analysis.py` | Complexity–entropy plane of 2x2 ordinal patterns, generated vs real held-out crops (`test_ordinal.py` tests them without a GPU) |
| `collect_results.py` | Turns all metric files into LaTeX tables/macros for the article |
| `article_plots.py` | Confusion-matrix figure in the article style, used by `collect_results.py` |
| `plot_nn_summary.py` | Summary figure of the nearest-neighbour check |
| `figures/` | Scripts for single article figures: U-Net diagram (Fig. 2), annotated textures (Fig. 4), confusion matrices |

## Order of execution (from `conditional/`)

```bash
python train_script.py                      # trains the model in config.json
python nll_calc.py                          # -> <save_checkpoint>/metrics/epoch_100/nll.json
python fid_real_reference.py               # -> <save_checkpoint>/metrics/fid_real_reference.json
python plot_fid.py --epoch 100              # -> .../epoch_100/fid_and_inception.json, fid_vs_num_steps.pdf
python resnet50/train_script.py             # classifier (once; reuses the same held-out files)
python resnet50/test_script.py              # -> resnet50/logs/resnet50/test/<run>/epoch_100/metrics.json
python nearest_neighbours.py                # memorisation check
python ordinal_analysis.py --epoch 100 --precision bf16   # -> <save_checkpoint>/metrics/epoch_100/ordinal/hc_plane.pdf
python collect_results.py --runs logs/lq256snr_cfg --names x0 --main x0 --epoch 100 --out ../results
```

## Prediction target

The default is clean-image prediction, `"prediction_type": "x0"` in `config.json`.
It is the target used in the article: physics-informed diffusion models apply
their constraints to the predicted clean image, so this network can be extended
in that direction directly. Noise prediction is available as an option:
set `"prediction_type": "eps"` (and a different `save_checkpoint`) and run the
same commands. With Min-SNR weighting both targets optimise the same objective.

## Fixed settings used in the article

* Sampling: DDIM with `num_steps = 100` and `eta = 1` (`config.json`), the global
  default. Every script that generates images (sample grids, FID, ResNet test,
  nearest neighbours, ordinal patterns) reads this value; 100 is also the
  fallback when the key is missing. `plot_fid.py` runs K = 10, 25, 50, 100 by
  default (`NUM_STEPS_VALUES`); `--steps` with no values gives only K = 100.
* FID protocol: every value uses 10,000 generated images against the same
  50,000 real crops; FID depends on the sample size, so it is compared only with
  the real-vs-real value computed at the same sizes (`fid_real_reference.py`).
* Guidance: `guidance_scale = 1.5` (w = 0.5), fixed (also the fallback when the key
  is missing; the NLL is always computed without guidance).
* Checkpoint: epoch 100 (`eval_epoch`). Longer training produced colour and structure
  leakage between classes.

## Choosing the evaluated epoch

All evaluation scripts load the checkpoint of `"eval_epoch"` in `config.json`
(default 100; `null` loads the latest one). Later epochs are not always better:
500 epochs gave worse samples than 100. Results are written per epoch, e.g.
`<save_checkpoint>/metrics/epoch_100/` and `resnet50/logs/resnet50/test/<run>/epoch_100/`,
so several epochs can be compared without overwriting. In `resnet50/config.json`,
`"denoiser_epoch"` overrides `eval_epoch` if set. Pass the same epoch to
`collect_results.py --epoch 100`.

## Conventions worth knowing when reading the code

* Timesteps are 0-indexed in code (`t = 0 … T-1`) and 1-indexed in the article.
* `guidance_scale = s` means `pred = uncond + s (cond − uncond)`; the article writes this
  as `(1 + w) cond − w uncond`, so `w = s − 1` (the default `s = 1.5` is `w = 0.5`).
* The null label for classifier-free guidance is the all-zeros vector.
* With `prediction_type = "x0"`, guidance is applied to the predicted clean image; since
  the conversion to ε is affine at fixed `x_t` and `t`, this is identical to guiding ε.
