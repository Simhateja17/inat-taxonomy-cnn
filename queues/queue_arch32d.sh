cd /Users/teja/python/ml
run() {
    seed=$1
    a=$2
    e=$3
    lv=$4
    suffix=$5
    out=results/all32_${a}_${lv}_s${seed}${suffix}.txt
    for attempt in 1 2 3; do
        .venv/bin/python experiments/single_arch.py ~/inat_data/inat_all32.npz $seed $lv $e $a > $out 2>&1
        if grep -q "Insufficient Memory" $out; then
            sleep 60
        else
            break
        fi
    done
}
for lv in family genus species; do
    run 1 alexnet 30 $lv ""
done
for pair in resnet:20 resnet18:15 resnet18_pre:8; do
    a=${pair%%:*}
    e=${pair##*:}
    for lv in kingdom phylum class order family genus species; do
        run 1 $a $e $lv ""
    done
done
for lv in kingdom phylum class order family genus species; do
    run 2 alexnet 30 $lv g
done
echo finished arch32d
