import sys
import numpy as np
import torch
npz=sys.argv[1]
seed=int(sys.argv[2])
level=sys.argv[3]
epochs=int(sys.argv[4])
torch.manual_seed(seed)

level_ids={'kingdom':1,'phylum':2,'class':3,'order':4,'family':5,'genus':6,'species':7}

d=np.load(npz)
y=torch.tensor(d['y']).long()
names=[]
for n in d['names']:
    names.append(str(n))
n_species=len(names)
n_photos=len(y)

groups=[]
for n in names:
    groups.append('_'.join(n.split('_')[1:level_ids[level]+1]))
group_names=sorted(set(groups))
numbers=[]
for g in groups:
    numbers.append(group_names.index(g))
species_group=torch.tensor(numbers)
n_groups=len(group_names)
labels=species_group[y]

position=torch.arange(n_photos)%50
train=position<30
val=(position>=30)&(position<40)
test=position>=40
X=torch.tensor(d['X']).permute(0,3,1,2).contiguous().float()/255
X=(X-0.5)/0.25
train_x,train_y=X[train],labels[train]
val_x,val_y=X[val],labels[val]
test_x,test_y=X[test],labels[test]
n_train=len(train_y)
print(level,n_groups,'groups',train_x.shape,val_x.shape,test_x.shape)

counts=torch.bincount(test_y,minlength=n_groups)
print('chance',1/n_groups,'biggest group share on test',counts.max().item()/len(test_y))

device='mps'
if len(sys.argv)>5 and sys.argv[5]=='cpu':
    device='cpu'
train_x,train_y=train_x.to(device),train_y.to(device)
val_x,val_y=val_x.to(device),val_y.to(device)
test_x,test_y=test_x.to(device),test_y.to(device)

perm=torch.randperm(n_train,device=device)
train_x=train_x[perm]
train_y=train_y[perm]

model=torch.nn.Sequential(
    torch.nn.Conv2d(3,32,3,padding=1), torch.nn.ReLU(),torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(32,64,3,padding=1), torch.nn.ReLU(),torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(64,128,3,padding=1), torch.nn.ReLU(),torch.nn.MaxPool2d(2),
    torch.nn.Flatten(),
    torch.nn.Linear(2048,256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(256,n_groups),
)
model=model.to(device)
print('parameters',sum(p.numel() for p in model.parameters()))

def scores_of(x):
    model.eval()
    out=[]
    with torch.no_grad():
        for i in range(0,len(x),1000):
            out.append(model(x[i:i+1000]))
    return torch.cat(out)

def accuracy(x,t):
    return (scores_of(x).argmax(dim=1)==t).float().mean().item()

loss_fn=torch.nn.CrossEntropyLoss()
opt=torch.optim.NAdam(model.parameters(),lr=0.0003)
for epoch in range(epochs):
    model.train()
    for i in range(0,n_train,32):
        xb=train_x[i:i+32]
        yb=train_y[i:i+32]
        if torch.rand(1).item()<0.5:
            xb=xb.flip(3)
        if torch.rand(1).item()<0.5:
            xb=xb.flip(2)
        loss=loss_fn(model(xb),yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(epoch,accuracy(train_x[:5000],train_y[:5000]),accuracy(val_x,val_y),accuracy(test_x,test_y),flush=True)

k=min(5,n_groups)
for name,x,t in [('train',train_x[:5000],train_y[:5000]),('val',val_x,val_y),('test',test_x,test_y)]:
    scores=scores_of(x)
    top1=(scores.argmax(dim=1)==t).float().mean().item()
    top5=(scores.topk(k,dim=1).indices==t[:,None]).any(dim=1).float().mean().item()
    print(name,'top1',top1,'top5',top5)
