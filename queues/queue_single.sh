cd /Users/teja/python/ml
for s in 1 2 3; do
    for lv in kingdom phylum class order family genus species; do
        .venv/bin/python experiments/single_level.py ~/inat_data/inat_all32.npz $s $lv 30 > results/all32_${lv}_s$s.txt 2>&1
    done
done
echo finished single
