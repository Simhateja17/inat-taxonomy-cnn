cd /Users/teja/python/ml
for lv in kingdom phylum class order family genus species; do
    .venv/bin/python kaggle/single_arch.py ~/inat_data/inat_all32.npz 1 $lv 30 lenet cpu > kaggle/results/all32_lenet_${lv}_s1.txt 2>&1
done
echo finished lenet cpu
