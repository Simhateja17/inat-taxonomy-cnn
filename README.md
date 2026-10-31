# inat-taxonomy-cnn

PyTorch experiments on the iNaturalist 2021 mini dataset: does knowing the taxonomy (kingdom, phylum, class, order, family, genus, species) help a small CNN?

Work in progress. A written report will be added here.

## What is in this repo
- `kaggle/` : all code, the saved result files (`kaggle/results/*.txt`, one file per run), the charts (`kaggle/figures/*.png`) and a running lab notebook (`kaggle/NOTES.md`).
- No photos are included. The dataset belongs to iNaturalist and has to be downloaded separately, for non-commercial research use only.

## Experiments
1. Baseline CNN on 200 species at 64x64, tuned on validation (flips, epochs, learning rate).
2. Two conditions on 200 species, 5 seeds: species head only vs species head plus an order head (multi-task), with shuffled-group controls.
3. The same on 500 species, 3 seeds.
4. Soft labels built from the taxonomy.
5. Six architectures (LeNet, our CNN, AlexNet-style, small ResNet, ResNet-18 style, ImageNet-pretrained ResNet-18).
6. All 2,076 species at 32x32, one network per taxonomy level.
7. CPU vs GPU.

See `kaggle/NOTES.md` for the setup and numbers of every experiment.

## How to run
The scripts expect to be run from the repo root, e.g. `python kaggle/taxo.py DATA.npz SEED LEVELS EPOCHS`. The `inat_pick*.py` scripts build the `.npz` files from the downloaded `train_mini` photos (paths inside them point to `~/inat_data`; change them to your own folder).
