import numpy as np

d=np.load('/Users/teja/inat_data/inat200_64.npz')
names=d['names']
orders = [str(n).split('_')[4] for n in names]
print(orders[:12])
print(len(set(orders)))
order_names = sorted(set(orders))
order_of_species = [order_names.index(o) for o in orders]
print(order_of_species[:12])