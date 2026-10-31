# Does telling a CNN the taxonomy help?

Small PyTorch CNNs on the iNaturalist 2021 mini dataset (photos of plants, animals and fungi). Every species sits in a tree of labels: kingdom, phylum, class, order, family, genus, species. This project asks two things:

1. If a small CNN is also told where each species sits in that tree, does it name the right species more often?
2. When it is wrong, is its wrong guess at least in the right group (the same class, order or family as the true species)?

**Full report:** [REPORT.pdf](REPORT.pdf) or [REPORT.md](REPORT.md). **Lab notebook** with every number, failure and decision: [NOTES.md](NOTES.md).

## Results at a glance

Main comparison: 200 species, 64x64 photos, 20 epochs, mean over 5 seeds.

| | condition 1: baseline (species head) | condition 2: baseline + order head |
|---|---|---|
| train error | 0.765 ± 0.023 | 0.776 ± 0.015 |
| validation error | 0.883 ± 0.003 | 0.882 ± 0.004 |
| test error (1 - accuracy) | 0.884 ± 0.003 | 0.877 ± 0.004 |
| test top-1 accuracy | 0.1162 | 0.1231 |
| wrong test guesses that land in the right order | 18.4% | 20.6% |
| same, with the order head's groups assigned at random (control) | not applicable | 18.4% |

![train, validation and test error of the two conditions](figures/two_conditions_errors.png)

- **The taxonomy changes how the model is wrong.** More of its wrong guesses land in the right order. The control with randomly assigned groups gives the baseline value (18.4%), so the effect needs the true taxonomy. It also appears with 500 species and with soft labels built from the taxonomy.
- **I cannot show that it changes how often the model is wrong.** The test gain of 0.007 is about two standard deviations, and validation shows no gain.
- **Pretraining gave the largest gain in test accuracy.** An ImageNet-pretrained ResNet-18 reaches 0.291 test top-1 against 0.116 for my baseline. More parameters trained from scratch did not raise it.
- **The finer the question, the harder it gets.** On 2,076 species (32x32 photos, one network per level), the test accuracy of my CNN falls from 0.85 at kingdom to 0.06 at species.

![share of wrong guesses in the right group](figures/right_group_share.png)

## What was run

| experiment | data | seeds |
|---|---|---|
| tuning the baseline (flips, epochs, learning rate) | 200 species | 5 |
| condition 1 against condition 2, with shuffled-group controls | 200 species | 5 |
| the same question on a larger set | 500 species | 3 |
| soft labels built from the taxonomy | 200 species | 5 |
| six architectures, including ImageNet pretraining | 200 species | 5 |
| which taxonomy levels are learned first | 200 species | 5 |
| one network per taxonomy level | 2,076 species, 32x32 | 1 to 3 |
| CPU against GPU | 200 and 400 species | 1 |

Every run is one file in `results/`. Section 14 of the report lists the limits and the failures.

## Repository layout

| folder | what is in it |
|---|---|
| `data_prep/` | scripts that build the `.npz` data files from the downloaded photos |
| `experiments/` | training scripts: `taxo.py` (the two conditions and controls), `softloss.py` (soft labels), `arch.py` (architectures), `single_level.py` and `single_arch.py` (one network per taxonomy level), `inat_cond2.py`, `cifar_torch.py`, `torch_toy.py` |
| `results/` | the printed output of every run, one file per run |
| `charts/` | scripts that make the figures from `results/`, and `summarize.py` |
| `figures/` | the charts used in the report |

## Data

The photos are not in this repository. The dataset belongs to iNaturalist and is for non-commercial research use. Download iNaturalist 2021 "Train Mini" yourself (Van Horn et al., *Benchmarking Representation Learning for Natural World Image Collections*, CVPR 2021). I used 2,076 complete species, 50 photos each. Section 3 of the report describes the dataset, the subsets and the train, validation and test split.

## Reproduce

Requirements: Python 3.9, PyTorch 2.8.0, torchvision 0.23.0, NumPy, Pillow and Matplotlib. I ran everything on a MacBook with an Apple M5 and 16 GB shared memory (PyTorch's `mps` GPU backend).

Run all commands from the repository root:

1. Point `data_prep/inat_pick.py`, `data_prep/inat_pick500.py` and `data_prep/inat_pick_all32.py` to your copy of the photos. They write the `.npz` files to `~/inat_data`.
2. Two conditions on 200 species: `python experiments/taxo.py ~/inat_data/inat200_64.npz SEED none 20` (baseline) and `python experiments/taxo.py ~/inat_data/inat200_64.npz SEED order 20` (order head). Add `shuffle` for the control.
3. Architectures: `python experiments/arch.py DATA SEED ARCH EPOCHS`. Soft labels: `python experiments/softloss.py DATA SEED BETA EPOCHS`.
4. One network per level on 2,076 species: `python experiments/single_level.py DATA SEED LEVEL EPOCHS` (my CNN) and `python experiments/single_arch.py DATA SEED LEVEL EPOCHS ARCH` (other networks).
5. Summaries: `python charts/summarize.py PREFIX CONFIG...`. Charts: `python charts/make_charts.py` and the other `charts/make_charts*.py` scripts.

## Author

Simha Teja ([@Simhateja17](https://github.com/Simhateja17)). Built in October 2026.
