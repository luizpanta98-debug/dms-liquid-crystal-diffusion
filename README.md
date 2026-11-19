# Generating polarised optical microscopy textures of liquid-crystal phases with class-conditional diffusion models

Code for the article by **Luiz F. M. Pantaleão and Rafael S. Zola**
(Department of Physics, State University of Maringá), submitted to *Soft Matter*.

A class-conditional denoising diffusion model (DDPM training, DDIM sampling,
classifier-free guidance, Min-SNR loss weighting) is trained on experimental
polarised optical microscopy (POM) textures of seven liquid-crystal classes
(nematic and smectic A, C, F, G, I) and evaluated with the likelihood bound, the
FID, a ResNet-50 phase classifier, the complexity–entropy plane of ordinal
patterns and a nearest-neighbour search in the training set.

## Repository structure

```
.
├── README.md            this file
├── requirements.txt     Python dependencies
└── conditional/         model, training, sampling and evaluation code
    ├── README.md        file-by-file description and order of execution
    ├── config.json      settings used in the article
    ├── resnet50/        phase classifier used for the phase-consistency test
    └── figures/         scripts for single article figures
```

## Installation

```bash
pip install -r requirements.txt
```

A CUDA GPU is needed for training and sampling. The article used one
NVIDIA RTX 4070 Ti SUPER.

## Data

The textures are not included in this repository. They belong to the
collection introduced by Osiecka-Drewniak *et al.* (*Liq. Cryst.* **51**, 128,
2024), in the version described by Voltarelli *et al.* (*Phys. Rev. E* **113**,
045414, 2026). The code expects one folder per class, next to `conditional/`:

```
data/
├── Nematic/   *.png
├── SmAa/
├── SmAb/
├── SmC/
├── SmF/
├── SmG/
└── SmI/
```

Each image is a 600 × 600 RGB texture. The path is set by `data_fp` in
`conditional/config.json` and `conditional/resnet50/config.json`.

## Usage

All commands run from `conditional/`; see `conditional/README.md` for the
complete order of execution.

```bash
cd conditional
python train_script.py        # trains the model of config.json
python test_script.py         # generates textures of every class
```

Trained weights, logs and metric files are written to `logs/` (not tracked).

## Citation

If you use this code, please cite the article (reference to be added on publication).
