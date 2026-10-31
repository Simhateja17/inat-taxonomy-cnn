# iNat project notes (for the final report)

Keep adding to this as we go. Facts and numbers only, with what was tried and what happened.

## Task
Professor's test: build simple PyTorch models on iNat 2021 mini (micro version allowed).
1) train and tune a baseline, 2) compare two experimental conditions, 3) summarise insights.
Report train, validation and test errors and other metrics. Deadline: Saturday 31 Oct 2026, 11:00 AM.

## Data decisions
- Source: iNat 2021 "Train Mini" (50 photos per species). Downloaded part of it: 2,076 complete species (103,800 photos, 8.9 GB).
- Picked 200 species: 20 random orders (from the 46 orders with at least 10 species) x 10 random species each. Seed 42.
- 50 photos per species -> 10,000 photos. Resized to 64x64 (squashed, not cropped). Saved to ~/inat_data/inat200_64.npz (not in git: dataset rules say do not distribute photos).
- Split per species by position in the folder: 30 train / 10 validation / 10 test = 6,000 / 2,000 / 2,000.
- Random guessing: 1/200 = 0.005 accuracy, loss ln(200) = 5.298.

## Warm-up results
- PyTorch CIFAR-10 CNN (same design as the Keras one, 10 epochs, no seeded comparison): validation 0.7164, test 0.7166. Keras version had 0.772 test at 20 epochs, so not a fair comparison yet.
- Speed on 6,000 photos at 64x64, one epoch: CPU 8.0 s, Apple GPU (mps) 4.2 s.

## Failure log (what went wrong, why, fix)
1. File named torch.py shadows the real torch package. Rename the file.
2. Typos: `nn.linear` (needs capital L), `torch.arrange` (is `arange`), `loss.backwards` (is `backward`).
3. Trailing comma after `CrossEntropyLoss()` made loss_fn a tuple -> "'tuple' object is not callable".
4. `opt.zero_grad` without brackets does nothing and gives no error.
5. Loss/optimizer lines placed inside `Sequential(...)` brackets -> NameError.
6. Training loop without `opt.step()`: loss stayed at 2.3167 (about ln 10 = 2.30) for all 1,407 batches. Weights never changed.
7. Block 3 without `padding=1`: output shrank to 1x1 instead of 2x2. Flatten size must match.
8. Training one batch repeatedly makes loss fall but says nothing about the dataset.
9. Data sorted by class (CIFAR and iNat): must shuffle before batches or the validation split.
10. **iNat baseline stuck at chance.** 3 conv blocks + Linear(8192,256) + Dropout 0.5 + Linear(256,200), NAdam lr 0.001, 20 epochs: validation accuracy 0.005 for every epoch. Loss about 5.30 = ln(200), the model predicted a single species for every photo.
    - Tried, 8 epochs each, still 0.005 every epoch: lr 0.001 raw pixels; lr 0.0001 raw pixels; lr 0.001 pixels centered ((x-0.5)/0.25); lr 0.0003 centered.
    - So learning rate and pixel centering alone are not the cause. 10 species and 50 species also stayed at chance, even on the training photos, on both CPU and GPU. That pointed to the data.
    - **Root cause: a loading bug.** In the photo-loading loop, `folder = os.path.join(root, picked[0])` used `picked[0]` (always the first species) instead of `name`. All 200 species got the same 50 photos: only 50 distinct photos out of 10,000, with labels 0 to 199 attached. No model can learn that.
    - Lesson: when a model cannot even fit its training data, check the data first. Quick check used: `len(np.unique(X.reshape(len(X), -1), axis=0))` gave 50 instead of 10,000.
    - After fixing the data (10,000 distinct photos) the same model was STILL stuck at 0.005 with lr 0.001. So there were two separate problems: the data bug, then a learning rate that was too high.
    - Tuning table, fixed data, 12 epochs, validation accuracy at the last epoch (chance 0.005). Single run each, seed 42, so differences of a few thousandths are noise:
      - lr 0.001, raw pixels: 0.005 (train 0.005), stuck
      - lr 0.001, centered pixels, batch norm: 0.005, stuck
      - lr 0.0001, centered pixels: 0.030 (train 0.059), still rising
      - lr 0.0001, centered pixels, batch norm: 0.045 (train 0.103), still rising
      - lr 0.0003, centered pixels: 0.093 (train 0.37), train far above val: overfitting begins
    - Centering alone, lr 0.0003, 12 epochs, same seed: raw pixels val 0.084 (train 0.137), centered val 0.093 (train 0.37). Centering made training faster, but the validation difference (0.009) is within what one run can show by chance. The learning rate was the big fix, not centering.
    - Takeaway: for this net and data, lr 0.001 with NAdam collapses to predicting one species; 0.0003 or lower learns. Pixel centering is `(x - 0.5) / 0.25`.
    - Typo seen: `X=(X-0.5)/2.5` (should be `/0.25`) squeezed pixels to about -0.2..0.2; it still learned but slower (val about 0.04 at epoch 19).
    - Seed check (centered, lr 0.0003, 12 epochs, `.contiguous()` after permute): seeds 1, 2, 3 gave val 0.079, 0.094, 0.083 (seed 42: 0.093). Spread across seeds is about 0.015. Any difference between two conditions smaller than that is not evidence.
    - **Memory layout matters on the GPU.** `.permute(0,3,1,2)` leaves the tensor non-contiguous (channels-last strides). Same code, same seed 1, 12 epochs: without `.contiguous()` val 0.022, with `.contiguous()` val 0.079. Cause not investigated (probably an Apple mps issue with such inputs). Always call `.contiguous()` after `permute` before training on mps.
11. Second typo in the save line: `names=np.array` (missing `(picked)`) saved the function itself, not the species names.

## Ideas / to try
- Easier sanity checks: 10 species only; overfit 64 photos.
- Batch normalisation, smaller classifier head, more epochs, bigger images.

## Baseline v1 result (Fri 30 Oct, 14:30)
Setup: 3 conv blocks (32, 64, 128 filters, 3x3, ReLU, maxpool), Linear(8192,256), ReLU, Dropout 0.5, Linear(256,200). 2,242,056 parameters. Pixels centered `(x-0.5)/0.25`, `.contiguous()` after permute, NAdam lr 0.0003, batch 32, 20 epochs, seed 42, Apple mps. Test photos not used yet.

| split (eval mode) | top-1 acc | top-5 acc |
|---|---|---|
| train (first 2,000 of 6,000, shuffled) | 0.785 | 0.931 |
| validation (2,000) | 0.0865 | 0.243 |

Chance: top-1 0.005, top-5 0.025. Validation top-1 stopped improving after about epoch 8 (0.086), while the training accuracy kept rising: strong overfitting. Single run.

Per-epoch run (second run of the same baseline, no seed change): training accuracy climbs 0.01 -> 0.80 over 20 epochs, validation peaks at 0.085 (epoch 12) and ends at 0.074. Final: train top-1 0.7995 / top-5 0.947, val top-1 0.074 / top-5 0.228.

### Does more training data help? (quick test, seed 1, 15 epochs, same net, lr 0.0003)
| training photos per species | train photos | final val top-1 |
|---|---|---|
| 10 | 2,000 | 0.037 |
| 20 | 4,000 | 0.058 |
| 30 | 6,000 | 0.086 |
Validation accuracy rises with more photos per species, so the overfitting is partly a lack-of-data problem. Single seed each; train accuracy column was noisy so not tabulated.

### More species instead of more photos? (side test, not finished)
200 random species from all 2,076 (30 train photos each, seed 1, 15 epochs): val top-1 0.104. Trying 400 species crashed twice with an Apple GPU out-of-memory error (`kIOGPUCommandBufferCallbackErrorOutOfMemory`), so there is no valid 400-species number. The question "does a larger species set raise accuracy" is untested. Dropped to save time.

### CPU vs GPU, and 200 vs 400 species (seed 1, 15 epochs, 30 train photos per species, NAdam lr 0.0003, centered pixels)
| test | GPU (mps) | CPU |
|---|---|---|
| our 200 species (20 orders x 10), val top-1 final | 0.086 | 0.075 (best epoch 0.086) |
| 200 random species, val top-1 | 0.104 | 0.119 |
| 400 random species, val top-1 | out of GPU memory | 0.082 (chance 0.0025, 32.8x chance) |
- CPU is not better than GPU: the two differences go in opposite directions and are the size of seed noise (about 0.015).
- 400 species: lower accuracy in absolute terms (0.082 vs 0.119 for 200), higher as a multiple of chance. Train accuracy was not measured, so this test does not say whether overfitting shrank. Each is a single run.

### Number of species vs overfitting (CPU, seed 1, 15 epochs, 30 train photos per species, nested random species lists)
| species | train photos | chance | train acc | val acc | gap | val as multiple of chance |
|---|---|---|---|---|---|---|
| 100 | 3,000 | 0.010 | 0.705 | 0.170 | 0.535 | 17x |
| 200 | 6,000 | 0.005 | 0.664 | 0.119 | 0.545 | 24x |
| 400 | 12,000 | 0.0025 | 0.539 | 0.082 | 0.457 | 33x |
More species: lower validation accuracy (harder task), gap stays large (about 0.5). The gap is a bit smaller at 400 mainly because train accuracy is lower (15 epochs, bigger task), not because validation improved. So more species did not reduce overfitting here. Single seed.

### Mistake severity, baseline v1 (third run of the same baseline on GPU; train top-1 0.810 / top-5 0.9505, val top-1 0.085 / top-5 0.239)
On the 2,000 validation photos the model made 1,830 wrong species guesses. In 292 of them (16.0%) the wrong guess was still in the correct order. A random wrong guess among the other 199 species would be in the right order 9/199 = 4.5% of the time, so the baseline is about 3.5x better than random at staying in the right order. This supports (does not prove) that similar-looking species inside one order get confused with each other. Note: the model's wrong guesses are not uniformly random, so 4.5% is a rough reference.
Run-to-run: the same code gave val top-1 0.0865, 0.074, 0.085 on three runs (no fixed seed on the model init order), so differences below about 0.015 are noise.

## Conditions to compare (plan)
- Condition 1 (baseline): predict species only.
- Condition 2: same CNN with an extra head predicting the order (20 orders), loss = species loss + order loss.
- Extra metric: mistake severity (is a wrong species at least in the right order).

## Condition 2 build (Fri 30 Oct, evening) - file `inat_cond2.py`
Setup: same body as the baseline (3 conv blocks, Flatten, Linear(8192,256), ReLU, Dropout 0.5), then two heads on the same 256 numbers `h`: `species_head` Linear(256,200) and `order_head` Linear(256,20). Loss = species cross-entropy + order cross-entropy (one number, so both heads and the shared body learn from it). Order labels: `order_of_species[y]` looks up the order number of each photo (20 orders, e.g. Rodentia = 16, Asparagales = 5). `train_order` must be shuffled with the same `perm` as the photos. NAdam lr 0.0003, batch 32, 20 epochs, seed 42, mps. Test set not used.

First complete run (single run), after fixing the severity line:
- train top-1 0.717, top-5 0.915; val top-1 0.085, top-5 0.248 (baseline runs: val 0.074 to 0.0865, top-5 0.228 to 0.243; difference is inside the 0.015 seed noise).
- Mistake severity: 1,830 wrong val guesses, 378 in the right order = 20.7% (baseline 16.0%, random reference 4.5%). One run each, so not yet evidence that the order head changes this. Need several seeds per condition.
- The order accuracy column printed during training was NOT valid in that run (see failure 14).

Second run, order accuracy fixed (compared with the order labels, photos fed into `body`), same code and seed 42, mps:
- val order accuracy (20 orders, chance 0.05): 0.134 at epoch 0, about 0.28 from epoch 9 on, 0.276 at epoch 19 (about 5.5x chance).
- train top-1 0.6625, top-5 0.8915; val top-1 0.0895, top-5 0.2535.
- Mistake severity: 1,821 wrong val guesses, 398 in the right order = 21.9%.
- Two Condition 2 runs so far: 20.7% and 21.9% in the right order, baseline 16.0% (one run). About 5 points higher, but not enough runs to know the noise. Next: several seeds for both conditions.

### Seeded comparison, validation only, 5 seeds each (seeds 1 to 5, 20 epochs, same code except the order head)
Seeds are now set at the top of both scripts (`python kaggle/inat_data.py SEED`, `python kaggle/inat_cond2.py SEED`). Same seed gives identical output (checked: condition 2 seed 1 run twice gave identical numbers on every epoch). The two conditions see the same photo shuffle for the same seed.

| seed | base val top-1 | base val top-5 | base right-order share | cond2 val top-1 | cond2 val top-5 | cond2 right-order share |
|---|---|---|---|---|---|---|
| 1 | 0.0845 | 0.2330 | 320/1831 = 17.5% | 0.0845 | 0.2525 | 377/1831 = 20.6% |
| 2 | 0.0935 | 0.2490 | 322/1813 = 17.8% | 0.0895 | 0.2670 | 395/1821 = 21.7% |
| 3 | 0.0855 | 0.2380 | 329/1829 = 18.0% | 0.1005 | 0.2785 | 392/1799 = 21.8% |
| 4 | 0.0900 | 0.2170 | 351/1820 = 19.3% | 0.0845 | 0.2500 | 383/1831 = 20.9% |
| 5 | 0.0740 | 0.2365 | 324/1852 = 17.5% | 0.0795 | 0.2515 | 374/1841 = 20.3% |
| mean | 0.0855 | 0.2347 | 18.0% | 0.0877 | 0.2599 | 21.1% |

Findings (validation, single split, 5 seeds):
- Right-order share: ranges do not overlap (baseline 17.5 to 19.3%, condition 2 20.3 to 21.8%). Condition 2 is about 3 points higher. The student predicted this before the runs.
- Val top-1: ranges overlap (0.074 to 0.0935 vs 0.0795 to 0.1005), means differ by 0.002. No evidence the order head changes species top-1.
- Val top-5: condition 2 higher on every mean and the ranges just touch (0.249 vs 0.250); weak but consistent, to recheck with more seeds.
- Train top-1 is similar or lower for condition 2 in most seeds (seed 1: 0.731 vs 0.790); both overfit hard (train 0.6 to 0.83 vs val about 0.09).
- Condition 2 validation order accuracy (20 orders, chance 0.05) ends at 0.27 to 0.30.
- Caveats: the first baseline value of 16.0% came from an unseeded run and sits below this range, so run-to-run spread is wider than the 5 seeds show. Test set still unused.

### Tuning step 1: random left-right flip (baseline, species head only, 5 seeds, 20 epochs)
Change in `inat_data.py`, inside the batch loop after `xb=train_x[i:i+32]`: `if torch.rand(1).item()<0.5: xb=xb.flip(3)`. Dimension 3 of `[32,3,64,64]` is the width, so about half of the batches are mirrored left to right. Labels are unchanged (a mirrored photo is the same species). Checked: 49 of 100 random draws were below 0.5.

| seed | train top-1 | val top-1 | val top-5 | right-order share |
|---|---|---|---|---|
| 1 | 0.4125 | 0.1140 | 0.3065 | 370/1772 = 20.9% |
| 2 | 0.3770 | 0.1130 | 0.2955 | 353/1774 = 19.9% |
| 3 | 0.3885 | 0.0985 | 0.2940 | 384/1803 = 21.3% |
| 4 | 0.4555 | 0.1045 | 0.2775 | 357/1791 = 19.9% |
| 5 | 0.4810 | 0.1060 | 0.2950 | 354/1788 = 19.8% |
| mean | 0.423 | 0.1072 | 0.2937 | 20.4% |

- Val top-1 ranges do not overlap with the no-flip baseline (no flip 0.074 to 0.0935, mean 0.0855; flip 0.0985 to 0.114, mean 0.1072). Val top-5 mean 0.235 -> 0.294. Student predicted all four new seeds would beat 0.0935; they did.
- Train-val gap fell from about 0.70 to about 0.32. Train accuracy is much lower (0.78 -> 0.42), so the network memorizes less. Train and val were still rising at epoch 19, so more epochs may help.
- **Confound to remember:** the baseline with flip has a right-order share of 19.8 to 21.3% (mean 20.4%), which overlaps condition 2 without flip (20.3 to 21.8%, mean 21.1%). So the earlier "condition 2 raises the right-order share" result may depend on how strong the species model is. The fair comparison is baseline+flip vs condition 2+flip on the same seeds. Condition 2 must get the flip too (labels `yb` and `ob` both unchanged).

### Condition 2 + flip (two heads, same random flip, 5 seeds, 20 epochs, validation only)
Flip added in `inat_cond2.py` after `ob=train_order[i:i+32]`; `yb` and `ob` are not flipped (a mirrored photo has the same species and the same order).

| seed | train top-1 | val top-1 | val top-5 | right-order share | val order acc |
|---|---|---|---|---|---|
| 1 | 0.3630 | 0.1005 | 0.2980 | 408/1799 = 22.7% | 0.3155 |
| 2 | 0.3510 | 0.1130 | 0.3245 | 400/1774 = 22.5% | 0.3225 |
| 3 | 0.3530 | 0.1255 | 0.3165 | 391/1749 = 22.4% | 0.3130 |
| 4 | 0.3830 | 0.1115 | 0.3150 | 410/1777 = 23.1% | 0.3160 |
| 5 | 0.4655 | 0.1145 | 0.3100 | 398/1771 = 22.5% | 0.3245 |
| mean | 0.383 | 0.1130 | 0.3128 | 22.6% | 0.318 |

Four-way view (means of 5 seeds, validation):

| | val top-1 | val top-5 | right-order share | train top-1 |
|---|---|---|---|---|
| baseline | 0.0855 | 0.2347 | 18.0% | 0.783 |
| baseline + flip | 0.1072 | 0.2937 | 20.4% | 0.423 |
| condition 2 | 0.0877 | 0.2599 | 21.1% | 0.731 |
| condition 2 + flip | 0.1130 | 0.3128 | 22.6% | 0.383 |

- Right-order share, with flip on both: baseline+flip 19.8 to 21.3% vs condition 2+flip 22.4 to 23.1%. Ranges do not overlap, so the order head raises the share of wrong guesses that stay in the right order (about +2.2 points, mean). The student predicted all four new seeds would stay above 21.3%; they did.
- Val top-1: baseline+flip 0.0985 to 0.114 vs condition 2+flip 0.1005 to 0.1255. Ranges overlap; mean difference 0.006. No clear species top-1 gain from the order head.
- Val top-5: 0.2775 to 0.3065 vs 0.298 to 0.3245, ranges overlap slightly; mean +0.019.
- Train-val gap: baseline+flip 0.42 - 0.11 = 0.32, condition 2+flip 0.38 - 0.11 = 0.27. Still overfitting, but less than the no-flip models (about 0.65 to 0.70).
- Flip helps both conditions (val top-1 means 0.0855 -> 0.1072 and 0.0877 -> 0.1130).

### Tuning step 2: add a random up-down flip (baseline, species head only, 5 seeds, 20 epochs)
Student's idea: mix many kinds of augmentation to reduce overfitting; test one kind at a time so we can tell which helped. Code in `inat_data.py` after the left-right flip: `if torch.rand(1).item()<0.5: xb=xb.flip(2)` (dimension 2 = height). Two independent coins per batch: none, left-right, up-down, or both (both = 180 degree rotation).

| seed | train top-1 | val top-1 | val top-5 | right-order share |
|---|---|---|---|---|
| 1 | 0.2525 | 0.1160 | 0.3135 | 375/1768 = 21.2% |
| 2 | 0.2400 | 0.1215 | 0.3215 | 323/1757 = 18.4% |
| 3 | 0.2325 | 0.1165 | 0.2985 | 330/1767 = 18.7% |
| 4 | 0.2630 | 0.1255 | 0.3170 | 326/1749 = 18.6% |
| 5 | 0.2460 | 0.1160 | 0.3080 | 349/1768 = 19.7% |
| mean | 0.247 | 0.1191 | 0.3117 | 19.3% |

- Val top-1 ranges do not overlap with left-right flip only (0.0985 to 0.114, mean 0.1072) vs both flips (0.116 to 0.1255, mean 0.1191), but the gap between ranges is only 0.002. Val top-5 ranges overlap (means 0.2937 vs 0.3117).
- Train-val gap fell to about 0.13 (train 0.247, val 0.119). Train accuracy is low and probably still rising at epoch 19, so more epochs should be tried next.
- Right-order share for the baseline fell to 18.4 to 21.2% (mean 19.3%, was 20.4% with left-right flip only); ranges overlap, so no clear change. This share moves around with the training setup.

### Tuning step 3: more epochs (baseline + both flips, 50 epochs instead of 20, 5 seeds)
Only `range(20)` changed to `range(50)` in `inat_data.py`. The student set 50 (not the 40 suggested). Prediction before the run: "still mostly learning". Only partly right.

| seed | train top-1 | val top-1 | val top-5 | right-order share |
|---|---|---|---|---|
| 1 | 0.6585 | 0.1200 | 0.3265 | 383/1760 = 21.8% |
| 2 | 0.6245 | 0.1190 | 0.3400 | 382/1762 = 21.7% |
| 3 | 0.6120 | 0.1325 | 0.3285 | 379/1735 = 21.8% |
| 4 | 0.6620 | 0.1195 | 0.3140 | 358/1761 = 20.3% |
| 5 | 0.6525 | 0.1235 | 0.3310 | 381/1753 = 21.7% |
| mean | 0.642 | 0.1229 | 0.3280 | 21.5% |

- 20 vs 50 epochs (both flips): train 0.247 -> 0.642; val top-1 mean 0.1191 -> 0.1229, ranges overlap (0.116 to 0.1255 vs 0.119 to 0.1325); val top-5 mean 0.3117 -> 0.3280, ranges overlap. Train-val gap 0.13 -> 0.52.
- Reading: after 20 epochs the extra epochs mostly add memorization, with at best a small val gain. 2.5x the time for no clear gain. Only the final-epoch numbers were saved (`tail -5`), so we do not know where val peaks; a per-epoch print for one seed would show it.
- Right-order share for the baseline rose back to 20.3 to 21.8% (mean 21.5%), close to condition 2 at 20 epochs. Again the share follows how well the species head is trained, so the condition 2 comparison must use identical settings for both conditions.
- Decision (student): use 20 epochs for the final runs. 50 epochs took more than twice as long for a val top-1 gain of 0.004 that is inside the seed noise.

### Condition 2 + both flips, 20 epochs (5 seeds, validation only)
Same settings as the tuned baseline (left-right and up-down flip, 20 epochs, NAdam 0.0003, batch 32), plus the order head.

| seed | train top-1 | val top-1 | val top-5 | right-order share | val order acc |
|---|---|---|---|---|---|
| 1 | 0.2200 | 0.1185 | 0.3195 | 384/1763 = 21.8% | 0.3255 |
| 2 | 0.2035 | 0.1155 | 0.3040 | 329/1769 = 18.6% | 0.3110 |
| 3 | 0.2280 | 0.1240 | 0.3445 | 379/1752 = 21.6% | 0.3330 |
| 4 | 0.2335 | 0.1155 | 0.3170 | 405/1769 = 22.9% | 0.3190 |
| 5 | 0.2475 | 0.1130 | 0.3180 | 406/1774 = 22.9% | 0.3350 |
| mean | 0.2265 | 0.1173 | 0.3206 | 21.6% | 0.325 |

Fair comparison, both conditions with both flips, 20 epochs, 5 seeds each (validation):

| | val top-1 | val top-5 | right-order share | train top-1 |
|---|---|---|---|---|
| baseline + 2 flips | 0.116 to 0.1255 (mean 0.1191) | 0.2985 to 0.3215 (0.3117) | 18.4 to 21.2% (19.3%) | 0.247 |
| condition 2 + 2 flips | 0.113 to 0.124 (mean 0.1173) | 0.304 to 0.3445 (0.3206) | 18.6 to 22.9% (21.6%) | 0.2265 |

- Species val top-1: no difference (ranges overlap, condition 2 mean is 0.002 lower).
- Val top-5: ranges overlap; condition 2 mean +0.009.
- Right-order share: mean +2.3 points for condition 2, higher on all five seeds when matched by seed number (+0.6, +0.2, +3.0, +4.3, +3.1), but the ranges now overlap (seed 2 of condition 2 is 18.6%, inside the baseline range 18.4 to 21.2%). With the strong augmentation the earlier clear gap (20.4% vs 22.6% with the left-right flip only, no overlap) shrank. Treat as a small, not clearly separated effect; 5 seeds are too few to settle it. Seeds are matched by number only, not truly paired (condition 2 draws extra random numbers for the second head).

### TEST SET results (first use of the test set; settings were chosen on validation only: both flips, 20 epochs)
Test photos: 2,000 (10 per species, never used for training or choosing settings). Test accuracy is printed once per run in the final loop (and per epoch, for the learning curve; no setting was chosen from it). Chance: top-1 0.005, top-5 0.025.

| seed | baseline test top-1 | baseline test top-5 | condition 2 test top-1 | condition 2 test top-5 |
|---|---|---|---|---|
| 1 | 0.1220 | 0.2975 | 0.1125 | 0.3055 |
| 2 | 0.1170 | 0.3035 | 0.1145 | 0.3045 |
| 3 | 0.1160 | 0.2890 | 0.1290 | 0.3160 |
| 4 | 0.1290 | 0.3105 | 0.1235 | 0.3140 |
| 5 | 0.1175 | 0.3130 | 0.1160 | 0.3105 |
| mean | 0.1203 | 0.3027 | 0.1191 | 0.3101 |

- Test top-1 error (1 - accuracy): baseline 0.880, condition 2 0.881. Test top-5 error: 0.697 and 0.690.
- Test means are close to validation means (baseline val 0.1191 / 0.3117; condition 2 val 0.1173 / 0.3206), so choosing settings on validation did not inflate the numbers.
- Ranges overlap on test top-1 and top-5: no clear species-accuracy difference between the conditions.
- Mistake severity on the TEST set (both scripts changed to use `test_x`, `test_y`; species test top-1 and top-5 reproduced the earlier numbers exactly):

| seed | baseline wrong guesses | baseline right order | share | condition 2 wrong guesses | condition 2 right order | share |
|---|---|---|---|---|---|---|
| 1 | 1756 | 315 | 17.9% | 1775 | 362 | 20.4% |
| 2 | 1766 | 327 | 18.5% | 1771 | 343 | 19.4% |
| 3 | 1768 | 333 | 18.8% | 1742 | 351 | 20.2% |
| 4 | 1742 | 332 | 19.1% | 1753 | 371 | 21.2% |
| 5 | 1765 | 298 | 16.9% | 1768 | 372 | 21.0% |
| mean | | | 18.3% | | | 20.4% |

- Test ranges do not overlap: baseline 16.9 to 19.1%, condition 2 19.4 to 21.2%, mean difference +2.1 points. Random reference: 9/199 = 4.5%. Student predicted the test shares would be similar to validation (19.3% and 21.6%); they were (18.3% and 20.4%).
- Summary for the report: the order head did not improve species top-1 or top-5 (ranges overlap), but it did make the model's mistakes land in the right order more often (+2 points, not overlapping on validation with the left-right flip, and not overlapping on test with both flips; on validation with both flips the ranges overlapped slightly).

## Taxonomy levels experiment (new script `taxo.py`, 200 species, 20 epochs, both flips, 5 seeds, test set)
One script for every condition: `python kaggle/taxo.py DATA.npz SEED LEVELS EPOCHS [shuffle] [cpu]`. LEVELS is `none` or a comma list of taxonomy levels; each level adds a Linear(256, number of groups) head, and the loss is the species loss plus one loss per extra head. Same body, same flips, NAdam 0.0003, batch 32. Evaluation in chunks of 500 photos (needed for 500 species on the GPU). Full outputs, including per-epoch train/val/test accuracy, are saved in `kaggle/results/taxo200_<levels>_s<seed>.txt`. Group counts in the 200 species: kingdom 3, phylum 4, class 8, order 20, family 77, genus 171 (genus is almost one group per species, so a genus head adds little).

| condition | test top-1 mean (range) | test top-5 mean | train top-1 mean | right class | right order | right family |
|---|---|---|---|---|---|---|
| baseline (species only) | 0.1162 (0.113 to 0.119) | 0.2967 | 0.235 | 46.5% | 18.4% (17.2 to 19.6) | 9.7% (8.7 to 10.7) |
| + order head | 0.1231 (0.119 to 0.128) | 0.3098 | 0.224 | 49.1% | 20.6% (19.0 to 22.0) | 10.6% (9.5 to 11.4) |
| + class, order, family heads | 0.1143 (0.107 to 0.119) | 0.3072 | 0.189 | 50.3% | 21.4% (20.1 to 22.4) | 11.6% (10.9 to 12.0) |

- Species top-1: the order head is slightly higher and its range only touches the baseline's; adding class and family heads gives no gain (0.1143).
- Mistake severity rises with more taxonomy heads at every level. Clear separation: class+order+family vs baseline on the family measure (10.9 to 12.0% vs 8.7 to 10.7%) and on the order measure (20.1 to 22.4% vs 17.2 to 19.6%). The order-head-only ranges overlap the baseline's slightly (order 19.0 vs 19.6).
- The order-head means agree with the earlier scripts (20.6% vs 20.4% for the order head, 18.4% vs 18.3% for the baseline on test), so the result repeats across code versions.
- Train accuracy falls as heads are added (0.235, 0.224, 0.189): the extra heads act like a regulariser. This is why a shuffled-taxonomy control is needed (see below).

### Shuffled-taxonomy control, order head (200 species, 20 epochs, both flips, 5 seeds, test set)
`taxo.py ... order 20 shuffle cpu`: the extra head still has 20 groups with the same group sizes (10 species each), but species are assigned to groups at random (fixed per seed, generator seed 1000+seed). Mistake severity is still measured with the REAL orders, families and classes. Summary script: `python kaggle/summarize.py taxo200 none order order-shuffle`.

| | test top-1 mean (range) | test top-5 mean | train top-1 | right class | right order (range) | right family (range) |
|---|---|---|---|---|---|---|
| baseline | 0.1162 (0.113 to 0.119) | 0.2967 | 0.235 | 46.5% | 18.4% (17.2 to 19.6) | 9.7% (8.7 to 10.7) |
| real order head | 0.1231 (0.119 to 0.128) | 0.3098 | 0.224 | 49.1% | 20.6% (19.0 to 22.0) | 10.6% (9.5 to 11.4) |
| shuffled order head | 0.1201 (0.118 to 0.122) | 0.2999 | 0.251 | 46.6% | 18.4% (16.6 to 19.4) | 9.6% (8.9 to 10.4) |

- Mistake severity: the shuffled head gives exactly the baseline share (18.4% right order, 9.6% right family); the real order head gives 20.6% / 10.6%. The ranges of real vs shuffled overlap slightly (19.0 vs 19.4 on order) but the means differ by 2.2 points and the shuffled mean equals the baseline. So the severity gain needs the real taxonomy; it is not just an extra head.
- Species top-1: shuffled 0.1201 lies between baseline (0.1162) and real (0.1231); ranges overlap with both. A small part of any accuracy gain may be regularisation from the extra task.
- Student's prediction before the run: the shuffled head would confuse the network. Result: it did not lower accuracy, but it gave no severity gain.

### Shuffled-taxonomy control, class + order + family heads (200 species, 20 epochs, both flips, 5 seeds, test set)
Same group sizes (8, 20, 77), species assigned to groups at random; severity measured with the real tree. `python kaggle/summarize.py taxo200 class-order-family class-order-family-shuffle`.

| | test top-1 (range) | test top-5 (range) | train top-1 | right class | right order | right family |
|---|---|---|---|---|---|---|
| baseline | 0.1162 (0.113 to 0.119) | 0.2967 (0.2905 to 0.3095) | 0.235 | 46.5% | 18.4% (17.2 to 19.6) | 9.7% (8.7 to 10.7) |
| real class+order+family | 0.1143 (0.107 to 0.119) | 0.3072 (0.294 to 0.315) | 0.189 | 50.3% (48.4 to 51.4) | 21.4% (20.1 to 22.4) | 11.6% (10.9 to 12.0) |
| shuffled class+order+family | 0.1106 (0.107 to 0.118) | 0.2869 (0.273 to 0.308) | 0.208 | 45.8% (44.9 to 46.9) | 18.1% (15.9 to 19.0) | 9.2% (7.8 to 10.1) |

- Real vs shuffled, ranges do not overlap on class (48.4 to 51.4 vs 44.9 to 46.9), order (20.1 to 22.4 vs 15.9 to 19.0) and family (10.9 to 12.0 vs 7.8 to 10.1). The shuffled heads give the baseline's severity (45.8 / 18.1 / 9.2% vs 46.5 / 18.4 / 9.7%). So the better mistakes come from the real taxonomy, not from having extra heads.
- Shuffled groups were slightly worse on species accuracy than the baseline (test top-1 0.1106 vs 0.1162, top-5 0.2869 vs 0.2967) but the ranges overlap, so only a hint that wrong groups confuse the network (the student's prediction).

## Overnight queue (started Sat 31 Oct, night) - results to read in the morning
`kaggle/run_overnight.sh` waits until no `taxo.py` run is left, then runs, one after another (200 species, 20 epochs, both flips, seeds 1 to 5, same data and splits as `taxo.py`):
1. `softloss.py` (soft labels from the taxonomy, idea of "Making Better Mistakes"): training target = weights over all species, `exp(-beta * d / 7)` normalised, where d = 0 for the true species and otherwise 1 + the number of levels (genus, family, order, class, phylum, kingdom) at which the two species differ. beta 7, 10 and 14 (beta 10 puts 0.829 of the weight on the true species; higher beta = closer to the normal loss). Our CNN, species head only. Files `soft200_beta<B>_s<seed>.txt`; summary: `python kaggle/summarize.py soft200 beta7 beta10 beta14`.
2. `arch.py` (species head only, same loop): `lenet` (354,636 parameters), `alexnet` (AlexNet-style for 64x64, 8,248,136 parameters, dropout), `resnet` (small residual net with batch norm, 4 residual blocks, 1,277,800 parameters); our CNN is the baseline row (2,242,056 parameters, `taxo200_none`). Files `arch200_<arch>_s<seed>.txt`; summary: `python kaggle/summarize.py arch200 lenet alexnet resnet`.
3. `resnet18` (added at the student's request: a ResNet of about 10 million weights): ResNet-18 layout (stem 3x3 stride 2, then 8 residual blocks with 64, 128, 256, 512 channels, global average pooling, dropout 0.5, linear head), 11,271,432 parameters. One epoch took about 28 s including loading. Queued by `queue_resnet18.sh` after the first queue finishes (`overnight2.log` shows `finished resnet18`). Files `arch200_resnet18_s<seed>.txt`; summary: `python kaggle/summarize.py arch200 resnet18`.
4. Learning order of the taxonomy levels (idea from Saxe, McClelland and Ganguli: broad distinctions are learned before fine ones). `arch.py` now also prints, after every epoch, a line `lvl EPOCH class X order Y family Z`: validation accuracy of the group guess, where the group probability is the sum of the species softmax probabilities inside the group (matrix product of the probabilities with a species-to-group 0/1 matrix) and the group guess is the group with the highest probability. Species accuracy per epoch is in the normal epoch line (3rd number = test, 2nd = validation, 1st = train). Every `arch200_*` run has these lines; `queue_levels.sh` additionally runs our CNN through `arch.py` (`arch200_cnn_s1..5.txt`, `overnight3.log` shows `finished cnn`). 3-epoch smoke test (seed 99, validation): class 0.42, 0.46, 0.46; order 0.11, 0.14, 0.19; family 0.08, 0.11, 0.14; species 0.014, 0.026, 0.037 (class guesses are not 1/8 at chance because the classes have very different sizes, so compare with the biggest-class share before drawing conclusions).
5. ImageNet fine-tuning (student's idea, as an extra experiment; the professor's wording "fine-tune the model and training process" most likely means tuning the settings, which we did): `arch.py ... resnet18_pre` = torchvision ResNet-18 with ImageNet weights (`IMAGENET1K_V1`, 44.7 MB downloaded from download.pytorch.org), new linear head for our species, photos doubled from 64x64 to 128x128 by bilinear upsampling inside the model, 11,279,112 parameters, same optimiser (NAdam 0.0003), flips, 20 epochs. 2-epoch smoke test (seed 99): test top-1 0.2515 (our from-scratch CNN gets about 0.12 after 20 epochs), about 30 s per epoch. Queued by `queue_pre.sh` (`overnight4.log` shows `finished pretrained`), files `arch200_resnet18_pre_s<seed>.txt`. This mixes in knowledge from 1.2 million ImageNet photos (which include many animals and plants), so report it as a separate experiment, not as one of the from-scratch conditions.
VGG16/VGG19 skipped: about 138 million weights, built for 224x224 photos, too slow on this Mac and certain to overfit on 6,000 photos.
Also still running when the queue was written: the 500-species job (`taxo500_*`, 3 seeds, 30 epochs: summary `python kaggle/summarize.py taxo500 none order class-order-family --seeds 3`) and the shuffled class+order+family control (`taxo200_class-order-family-shuffle_*`).
One-epoch smoke tests only showed the scripts run; the AlexNet-style net was still at chance after epoch 0 (train 0.0045), so it may stall at lr 0.0003 like our first baseline did at lr 0.001; if so, report it as is.

### More failures (Condition 2 build)
12. Removed `model` when splitting into `body` + two heads, but the optimizer and loop still said `model` -> `NameError`. Every use of `model(xb)` becomes `species_head(body(xb))` (species) or `order_head(body(xb))` (order).
13. `order_names.append(o)` instead of `order_names.index(o)`: `.append` changes the list and returns `None`, so the list of order numbers filled with `None`.
14. Order accuracy compared against the species labels (`val_y`, `train_y`) instead of the order labels (`val_order`, `train_order`). The order scores have 20 classes and the species labels run 0 to 199, so the printed number stayed near 0.01 and means nothing. Fix: compare with `val_order`.
15. `species_head().train()`: brackets call the layer with no input. Heads have no dropout, so only `body.train()` is needed.
16. **`same_order=order_of_species[guess]=order_of_species[val_y]`** (single `=` instead of `==`): this assigns instead of compares. It overwrote entries of `order_of_species` and made `same_order` a list of order numbers, so `wrong & same_order` counted wrong guesses with an odd order number. It printed 888 "wrong but right order" (48.8%), a meaningless number that looked like a big win. After fixing to `==` the count was 378 (20.7%). Lesson: a result that looks too good right after new code is a reason to re-read the code first.

## 500 species (`inat500_64.npz`: 25 orders x 20 random species, 25,000 photos, 15,000 / 5,000 / 5,000; 30 epochs, both flips, 3 seeds, test set)
Built by `inat_pick500.py` (random seed 42, from the 26 orders with at least 20 species on disk; 24,999 distinct photos, one duplicate in the source data). 11 classes, 25 orders, 177 families in this set; chance top-1 0.002. `python kaggle/summarize.py taxo500 none order class-order-family --seeds 3`.

| condition | test top-1 mean (range) | test top-5 mean (range) | train top-1 | right class | right order (range) | right family |
|---|---|---|---|---|---|---|
| baseline | 0.1105 (0.104 to 0.114) | 0.2902 (0.276 to 0.298) | 0.309 | 45.8% | 19.7% (19.3 to 20.0) | 6.1% |
| + order head | 0.1158 (0.113 to 0.119) | 0.3049 (0.299 to 0.311) | 0.316 | 48.2% | 21.9% (21.7 to 22.0) | 6.6% |
| + class, order, family heads | 0.1158 (0.111 to 0.119) | 0.3104 (0.308 to 0.315) | 0.274 | 49.8% | 22.8% (22.4 to 23.5) | 6.8% |

- With 2.5 times more species the baseline test top-1 is 0.1105 (about 52 times chance), only a little below the 200-species value of 0.1162; there are also 2.5 times more training photos in total.
- Right-order share rises with taxonomy heads (19.7% -> 21.9% -> 22.8%), ranges do not overlap. Test top-5 also rises (0.290 -> 0.305 -> 0.310) with ranges that just separate (0.298 vs 0.299 and 0.308), so with more species the extra heads help top-5 too. Top-1 ranges overlap. Only 3 seeds, so treat as a first indication.
- Right-family share is low (6%) because the 500 species sit in 177 families.

## Overnight results (200 species, 20 epochs, both flips, seeds 1 to 5, test set; all 45 runs finished)
Summaries: `python kaggle/summarize.py arch200 lenet alexnet cnn resnet resnet18 resnet18_pre` and `python kaggle/summarize.py soft200 beta7 beta10 beta14`.

### Architectures (species head only)
| architecture | parameters | train top-1 | test top-1 mean (range) | test top-5 mean (range) | right order | right family |
|---|---|---|---|---|---|---|
| LeNet | 354,636 | 0.112 | 0.0665 (0.060 to 0.072) | 0.2018 (0.190 to 0.220) | 14.5% | 7.0% |
| AlexNet-style | 8,248,136 | 0.172 | 0.1066 (0.097 to 0.116) | 0.2835 (0.269 to 0.292) | 18.5% | 9.4% |
| our CNN (baseline) | 2,242,056 | 0.235 | 0.1162 (0.113 to 0.119) | 0.2967 (0.290 to 0.310) | 18.4% | 9.7% |
| small ResNet | 1,277,800 | 0.161 | 0.1124 (0.101 to 0.119) | 0.2930 (0.282 to 0.307) | 18.2% | 9.8% |
| ResNet-18 style, from scratch | 11,271,432 | 0.261 | 0.1146 (0.103 to 0.125) | 0.2974 (0.284 to 0.315) | 18.6% | 9.7% |
| ResNet-18, ImageNet pretrained | 11,279,112 | 0.852 | 0.2914 (0.285 to 0.303) | 0.5584 (0.535 to 0.577) | 29.5% | 16.3% |

- From scratch, only LeNet is clearly worse (too small: train 0.112, underfits). AlexNet-style, our CNN, small ResNet and ResNet-18 style all land at 0.107 to 0.116 with overlapping ranges: more parameters (1.3 million to 11.3 million) did not help with 6,000 training photos.
- ImageNet pretraining more than doubles test top-1 (0.291 vs 0.116) and top-5 (0.558 vs 0.297), ranges far apart; its mistakes are also smarter (right order 29.5% vs 18.4%, right family 16.3% vs 9.7%). But train 0.852 vs test 0.291 is still a huge gap; per-epoch validation species accuracy peaks around epoch 10 (0.306) and falls to 0.287 at epoch 20, so early stopping would help.

### Soft labels from the taxonomy (our CNN, species head only; beta = how sharply the target concentrates on the true species)
| beta | weight on the true species (seed-independent) | train top-1 | test top-1 mean (range) | test top-5 mean | right class | right order (range) | right family (range) |
|---|---|---|---|---|---|---|---|
| none (baseline) | 1.0 | 0.235 | 0.1162 (0.113 to 0.119) | 0.2967 | 46.5% | 18.4% (17.2 to 19.6) | 9.7% (8.7 to 10.7) |
| 14 | high | 0.233 | 0.1196 (0.109 to 0.129) | 0.3049 | 46.5% | 19.0% (18.5 to 19.5) | 10.1% (9.1 to 10.4) |
| 10 | 0.829 | 0.227 | 0.1177 (0.107 to 0.124) | 0.3106 | 47.2% | 20.6% (19.6 to 21.2) | 10.6% (9.5 to 11.5) |
| 7 | lower | 0.190 | 0.1070 (0.100 to 0.116) | 0.2931 | 49.2% | 22.2% (21.9 to 22.8) | 11.8% (11.3 to 12.4) |
- Lower beta (softer targets) gives smarter mistakes without any extra head: beta 7 reaches right order 22.2% and right family 11.8%, as good as the three-head model (21.4% and 11.6%), with ranges clear of the baseline. The price is exact accuracy: test top-1 falls to 0.1070 (range overlaps the baseline). beta 10 and 14 keep top-1 at about 0.118 with a smaller severity gain. A dial between exact accuracy and mistake quality.

### Learning order of the levels (validation accuracy of the group guess from summed softmax probabilities, mean of 5 seeds)
Biggest-group baselines: class 0.35, order 0.05, family 0.05; species chance 0.005.
| model | epoch | class | order | family | species |
|---|---|---|---|---|---|
| our CNN | 1 | 0.399 | 0.094 | 0.081 | 0.013 |
| our CNN | 5 | 0.489 | 0.212 | 0.160 | 0.050 |
| our CNN | 10 | 0.542 | 0.264 | 0.199 | 0.082 |
| our CNN | 20 | 0.584 | 0.309 | 0.234 | 0.117 |
| LeNet (2 conv layers) | 20 | 0.518 | 0.223 | 0.164 | 0.061 |
| ResNet-18 style | 20 | 0.575 | 0.306 | 0.226 | 0.110 |
| ResNet-18 pretrained | 1 | 0.694 | 0.453 | 0.339 | 0.185 |
- In every model coarse groups are always more accurate than fine ones (class > order > family > species), and all levels rise together smoothly; we see no clear stage-like jumps (coarse first, then fine) and no clear difference between the shallow LeNet and the deeper nets. The class level starts near the biggest-class baseline (0.40 vs 0.35) and gains little (0.58 at epoch 20), while species gains most relative to chance (0.013 -> 0.117). The pretrained network learns all levels within the first epoch.

## Full dataset at 32x32, one network per taxonomy level (started Sat 31 Oct, morning)
Student's plan: use all 2,076 species and every photo, shrunk to 32x32 so the results can be compared with CIFAR-10 (32x32), then train one network per taxonomy level from the left (kingdom) to the right (species) to see how accuracy changes as the question gets finer.
- Data: `inat_pick_all32.py` -> `~/inat_data/inat_all32.npz` (103,800 photos, 2,076 species, 50 per species, 320 MB, not in git). Split per species by position: 30 train / 10 val / 10 test = 62,280 / 20,760 / 20,760.
- Groups in the full set: kingdom 3 (biggest group = 53.6% of species), phylum 11 (42.7%), class 35 (33.2%), order 170 (15.0%), family 539 (5.3%), genus 1,566, species 2,076. Group = the whole path from the kingdom down to that level.
- `single_level.py DATA SEED LEVEL EPOCHS`: our 3-block CNN adapted to 32x32 (Linear(2048,256) after the three poolings, dropout 0.5, one output per group), NAdam 0.0003, batch 32, both flips, evaluation in chunks. Each run prints per-epoch train (first 5,000 shuffled train photos) / val / test accuracy and final top-1 and top-5 (top-5 capped at the number of groups). Queue A: `queue_single.sh`, files `all32_<level>_s<seed>.txt`, 30 epochs, seeds 1 to 3 (`overnight5.log`).
- First finished run, kingdom, seed 1: test top-1 0.848 (chance 0.333, always guessing the biggest group 0.536). Validation peaked at about 0.8555 around epoch 15 and then slowly fell (0.851 at epoch 30) while train rose from 0.785 to 0.907: more epochs did not help even with 10 times more training photos.
- Other architectures on the same data: `single_arch.py DATA SEED LEVEL EPOCHS ARCH` with `lenet` (30 epochs), `alexnet` (30), `resnet` (20), `resnet18` (15, first layer with stride 2), `resnet18_pre` (8, ImageNet weights, photos doubled to 64x64). One seed each (seed 1). Queue B: `queue_arch32.sh`, waits for queue A, retries a run up to 3 times if the GPU reports `Insufficient Memory` (`overnight6.log`), files `all32_<arch>_<level>_s1.txt`.
- Failure log: while queue A was running, two extra test runs (AlexNet-style and ResNet-18 style) failed with `kIOGPUCommandBufferCallbackErrorOutOfMemory` and printed accuracies of 0.0; the 16 GB of the Mac are shared between the GPU and every other program, so test runs should not be started next to a training queue. The first ResNet-18 style version with a stride-1 first layer took about 165 s per epoch on 62,280 photos (too slow), so the first layer was changed to stride 2.

## Charts (all made from the saved result files, `kaggle/figures/*.png`)
Scripts: `make_charts.py` and `make_charts2.py` (run from the `ml` folder). Photos are never put into charts or the repo.
1. `right_group_share.png`: share of wrong guesses landing in the right order and the right family, one dot per seed, baseline vs shuffled controls vs real groups (200 species, test set).
2. `learning_curves.png`: train and test accuracy per epoch, baseline vs order head.
3. `architectures.png`: test top-1 per seed for LeNet, AlexNet-style, our CNN, small ResNet, ResNet-18 style and ResNet-18 pretrained.
4. `overfitting_gap.png`: train accuracy against test accuracy for each architecture.
5. `accuracy_vs_mistake_quality.png`: test top-1 against right-order share for baseline, extra heads and the three soft-label settings (mean and range).
6. `level_learning_curves.png`: validation accuracy per epoch at class, order, family and species level for LeNet, our CNN, ResNet-18 style and pretrained.
Script `make_charts3.py` (early tuning and the 500-species test):
7. `tuning_steps.png`: validation top-1 of the baseline, one dot per seed (5 seeds), no flip, left-right flip, both flips, 50 epochs. We kept both flips with 20 epochs.
8. `data_amount.png`: early single runs, validation accuracy against photos per species (10/20/30) and against number of species (100/200/400), with train accuracy for the second panel.
9. `species_500.png`: 500 species, test set, 3 seeds: top-1, top-5 and right-order share for baseline, order head and class+order+family heads.
Script `make_charts4.py` (full dataset, 32x32, one network per taxonomy level; reads `all32_*` files, so it can be rerun as runs finish):
10. `full32_top1_by_level.png`: test top-1 from kingdom to species for our CNN (mean of up to 3 seeds, dots = seeds), other architectures (1 seed), chance, always-guess-the-biggest-group, and the CIFAR-10 reference line 0.717.
11. `full32_top5_by_level.png`: the same with top-5.
12. `full32_curves_by_level.png`: train and validation accuracy per epoch for each level (our CNN).
Status when logged (2026-10-31): charts 10 to 12 contain only the runs finished so far (our CNN seed 1 for all 7 levels, seed 2 for kingdom and phylum, LeNet on CPU for kingdom, phylum, class). Test top-1 so far: kingdom 0.853, phylum 0.742, class 0.618, order 0.282, family 0.147, genus 0.074, species 0.059. Rerun `.venv/bin/python kaggle/make_charts4.py` when the queues finish.
Still to make: the shuffled class+order+family control is already in chart 1; nothing else pending except the final rerun of charts 10 to 12 and any new experiments.
To do at the end, after all GPU queues have finished (so no job is interrupted): the CPU vs GPU chart (from the table in the CPU vs GPU section), a rerun of `make_charts4.py` with all architectures (AlexNet seed 2 GPU files are named `all32_alexnet_<level>_s2g.txt`), then the report.
- AlexNet failure and fix (2026-10-31): AlexNet seed 1 at family and genus printed 0.0 everywhere with `kIOGPUCommandBufferCallbackErrorOutOfMemory`, also with the GPU idle and 7 GB free, so it was not a lack of memory. A bare AlexNet forward/backward/NAdam step passed for 170, 539 and 1566 groups, and the same script worked for kingdom to order. Likely cause (not proven): `single_arch.py` queued about 1,950 GPU steps per epoch without waiting, and AlexNet's heavy compute let the queue grow until the GPU buffers ran out. Fix: `loss.item()` every 100 steps (waits for the GPU, changes no numbers). With the fix the 2-epoch genus test ran (test top-1 0.0108). The failed family and genus files are being rerun by `queue_arch32d.sh` (log `overnight12.log`): AlexNet family, genus, species, then resnet, resnet18, resnet18_pre (seed 1), then AlexNet seed 2 on the GPU (files `_s2g`). Earlier "out of memory" test failures next to the queue may have had the same cause.

## Full dataset, 32x32, one network per taxonomy level: final table (2026-10-31, all queues finished 18:05)
Test top-1, mean over seeds (our CNN and LeNet 3 seeds, AlexNet-style 2 seeds, the rest seed 1). Epochs: CNN 30, LeNet 30, AlexNet 30, small ResNet 20, ResNet-18 style 15, ResNet-18 pretrained 8.
| network | kingdom | phylum | class | order | family | genus | species |
| our CNN | 0.852 | 0.743 | 0.616 | 0.283 | 0.147 | 0.077 | 0.060 |
| LeNet | 0.822 | 0.686 | 0.559 | 0.223 | 0.106 | 0.038 | 0.024 |
| AlexNet-style | 0.849 | 0.728 | 0.611 | 0.285 | 0.149 | 0.053 | 0.007 |
| small ResNet | 0.857 | 0.747 | 0.634 | 0.302 | 0.169 | 0.107 | 0.083 |
| ResNet-18 style | 0.852 | 0.738 | 0.611 | 0.286 | 0.157 | 0.081 | 0.056 |
| ResNet-18 pretrained | 0.874 | 0.774 | 0.654 | 0.309 | 0.189 | 0.112 | 0.093 |
- AlexNet-style at species: seed 1 collapsed to chance (test 0.0005, validation exactly 0.0005 at every epoch, the network predicted one species for every photo); seed 2 learned slowly (test 0.0140, top-5 0.0588). The 0.007 in the table is the mean of those two, so it hides the collapse.
- Parameters (kingdom / species output): LeNet 61,411 / 237,616; our CNN 618,563 / 1,151,324; small ResNet 1,227,171 / 1,759,932; AlexNet-style 3,428,483 / 4,491,932; ResNet-18 style 11,170,371 / 12,233,820; ResNet-18 pretrained 11,178,051 / 12,241,500.
- Time per level (GPU): AlexNet about 19 min (30 epochs, 38 s per epoch), small ResNet about 7 min (20 epochs), ResNet-18 style about 17 min (15 epochs, 67 s per epoch), pretrained about 10.6 min (8 epochs). A memory squeeze (swap about 3 GB, VS Code helper 2.3 GB) once made the ResNet-18 style epochs 3.5 times slower; restarting the VS Code extension host fixed it.
- Chart script `make_charts4.py` now draws all six networks and averages seeds (files `_s1`, `_s2`, `_s3`, AlexNet seed 2 as `_s2g`); value labels on the lines were removed because they overlapped.
- `make_charts5.py` makes chart 13, `cpu_vs_gpu.png`: left, seconds per epoch (CPU 8.0 s, Apple GPU 4.2 s, 6,000 photos at 64x64); right, validation top-1 for GPU and CPU on three tests (200 species 20 orders x 10: 0.086 vs 0.075; 200 random species: 0.104 vs 0.119; 400 random species: GPU out of memory, CPU 0.082). Single runs, seed noise about 0.015. The 400-species GPU "out of memory" may have had the same cause as the AlexNet failure (steps queued without waiting); not tested.
