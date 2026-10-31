cd /Users/teja/python/ml
while pgrep -f "run_overnight.sh" > /dev/null || pgrep -f "queue_resnet18.sh" > /dev/null || pgrep -f "queue_levels.sh" > /dev/null; do
    sleep 30
done
for s in 1 2 3 4 5; do
    .venv/bin/python experiments/arch.py ~/inat_data/inat200_64.npz $s resnet18_pre 20 > results/arch200_resnet18_pre_s$s.txt 2>&1
done
echo finished pretrained
