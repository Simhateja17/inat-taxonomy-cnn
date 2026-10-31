cd /Users/teja/python/ml
while pgrep -f "run_overnight.sh" > /dev/null; do
    sleep 30
done
for s in 1 2 3 4 5; do
    .venv/bin/python experiments/arch.py ~/inat_data/inat200_64.npz $s resnet18 20 > results/arch200_resnet18_s$s.txt 2>&1
done
echo finished resnet18
