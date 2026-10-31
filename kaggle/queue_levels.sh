cd /Users/teja/python/ml
while pgrep -f "run_overnight.sh" > /dev/null || pgrep -f "queue_resnet18.sh" > /dev/null; do
    sleep 30
done
for s in 1 2 3 4 5; do
    .venv/bin/python kaggle/arch.py ~/inat_data/inat200_64.npz $s cnn 20 > kaggle/results/arch200_cnn_s$s.txt 2>&1
done
echo finished cnn
