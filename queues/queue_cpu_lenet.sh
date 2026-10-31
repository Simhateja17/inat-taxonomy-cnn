cd /Users/teja/python/ml
for seed in 2 3; do
    for lv in kingdom phylum class order family genus species; do
        .venv/bin/python experiments/single_arch.py ~/inat_data/inat_all32.npz $seed $lv 30 lenet cpu > results/all32_lenet_${lv}_s${seed}.txt 2>&1
    done
done
echo finished lenet seeds
