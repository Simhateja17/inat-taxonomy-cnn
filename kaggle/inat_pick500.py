import os
import random
import numpy as np
from PIL import Image
root=os.path.expanduser('~/inat_data/train_mini')
names=sorted(os.listdir(root))

orders={}
for n in names:
    order=n.split('_')[4]
    orders.setdefault(order,[]).append(n)

big=[]
for o in orders:
    if len(orders[o])>=20:
        big.append(o)
print(len(names),len(orders),len(big))

random.seed(42)
chosen_orders=random.sample(sorted(big),25)
picked=[]
for o in chosen_orders:
    picked=picked+random.sample(orders[o],20)
print(len(picked))

images=[]
labels=[]
for label,name in enumerate(picked):
    folder=os.path.join(root,name)
    for f in sorted(os.listdir(folder)):
        img=Image.open(os.path.join(folder,f)).convert('RGB').resize((64,64))
        images.append(np.array(img))
        labels.append(label)
print(len(images),len(labels))
X=np.stack(images)
y=np.array(labels)
print(X.shape,y.shape)
print(len(np.unique(X.reshape(len(X),-1),axis=0)))
np.savez('/Users/teja/inat_data/inat500_64.npz',X=X,y=y,names=np.array(picked))
