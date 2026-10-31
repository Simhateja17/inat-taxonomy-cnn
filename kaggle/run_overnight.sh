cd /Users/teja/python/ml
while pgrep -f "kaggle/taxo.py" > /dev/null; do
    sleep 30
done
for beta in 7 10 14; do
    for s in 1 2 3 4 5; do
        .venv/bin/python kaggle/softloss.py ~/inat_data/inat200_64.npz $s $beta 20 > kaggle/results/soft200_beta${beta}_s$s.txt 2>&1
    done
done
for a in lenet alexnet resnet; do
    for s in 1 2 3 4 5; do
        .venv/bin/python kaggle/arch.py ~/inat_data/inat200_64.npz $s $a 20 > kaggle/results/arch200_${a}_s$s.txt 2>&1
    done
done
echo finished
