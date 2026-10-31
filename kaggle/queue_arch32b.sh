cd /Users/teja/python/ml
while pgrep -f "queue_single.sh" > /dev/null; do
    sleep 30
done
for pair in alexnet:30 resnet:20 resnet18:15 resnet18_pre:8; do
    a=${pair%%:*}
    e=${pair##*:}
    for lv in kingdom phylum class order family genus species; do
        out=kaggle/results/all32_${a}_${lv}_s1.txt
        for attempt in 1 2 3; do
            .venv/bin/python kaggle/single_arch.py ~/inat_data/inat_all32.npz 1 $lv $e $a > $out 2>&1
            if grep -q "Insufficient Memory" $out; then
                sleep 60
            else
                break
            fi
        done
    done
done
echo finished arch32b
