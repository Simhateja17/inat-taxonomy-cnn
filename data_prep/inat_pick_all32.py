import os
import numpy as np
from PIL import Image
root=os.path.expanduser('~/inat_data/train_mini')
names=sorted(os.listdir(root))
print(len(names))

images=[]
labels=[]
for label,name in enumerate(names):
    folder=os.path.join(root,name)
    for f in sorted(os.listdir(folder)):
        img=Image.open(os.path.join(folder,f)).convert('RGB').resize((32,32))
        images.append(np.array(img))
        labels.append(label)
    if label%200==0:
        print(label,len(images),flush=True)
X=np.stack(images)
y=np.array(labels)
print(X.shape,y.shape)
np.savez('/Users/teja/inat_data/inat_all32.npz',X=X,y=y,names=np.array(names))
print('saved')
