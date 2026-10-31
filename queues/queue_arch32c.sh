cd /Users/teja/python/ml
while pgrep -f "queue_arch32b.sh" > /dev/null; do
    sleep 30
done
for lv in kingdom phylum class order family genus species; do
    out=results/all32_alexnet_${lv}_s2g.txt
    for attempt in 1 2 3; do
        .venv/bin/python experiments/single_arch.py ~/inat_data/inat_all32.npz 2 $lv 30 alexnet > $out 2>&1
        if grep -q "Insufficient Memory" $out; then
            sleep 60
        else
            break
        fi
    done
done
echo finished arch32c
