import sys
import numpy as np
import torch
npz=sys.argv[1]
seed=int(sys.argv[2])
beta=float(sys.argv[3])
epochs=int(sys.argv[4])
torch.manual_seed(seed)

level_ids={'kingdom':1,'phylum':2,'class':3,'order':4,'family':5,'genus':6}

d=np.load(npz)
y=torch.tensor(d['y']).long()
names=[]
parts=[]
for n in d['names']:
    names.append(str(n))
    parts.append(str(n).split('_'))
n_species=len(names)
n_photos=len(y)

position=torch.arange(n_photos)%50
train=position<30
val=(position>=30)&(position<40)
test=position>=40
X=torch.tensor(d['X']).permute(0,3,1,2).contiguous().float()/255
X=(X-0.5)/0.25
train_x,train_y=X[train],y[train]
val_x,val_y=X[val],y[val]
test_x,test_y=X[test],y[test]
n_train=len(train_y)
print(n_species,'species',train_x.shape,val_x.shape,test_x.shape)

def group_of_species(level):
    groups=[]
    for n in names:
        groups.append(n.split('_')[level_ids[level]])
    group_names=sorted(set(groups))
    numbers=[]
    for g in groups:
        numbers.append(group_names.index(g))
    return torch.tensor(numbers)

sev_levels=['class','order','family']
species_group={}
for lv in sev_levels:
    species_group[lv]=group_of_species(lv)

dist=torch.zeros(n_species,n_species)
for i in range(n_species):
    for j in range(n_species):
        if i!=j:
            count=1
            for lv in [6,5,4,3,2,1]:
                if parts[i][lv]!=parts[j][lv]:
                    count=count+1
            dist[i,j]=count
soft=torch.exp(-beta*dist/7)
soft=soft/soft.sum(dim=1,keepdim=True)
print('beta',beta,'weight on the true species',soft[0,0].item())

device='mps'
if len(sys.argv)>5 and sys.argv[5]=='cpu':
    device='cpu'
train_x,train_y=train_x.to(device),train_y.to(device)
val_x,val_y=val_x.to(device),val_y.to(device)
test_x,test_y=test_x.to(device),test_y.to(device)
soft=soft.to(device)
for lv in species_group:
    species_group[lv]=species_group[lv].to(device)

perm=torch.randperm(n_train,device=device)
train_x=train_x[perm]
train_y=train_y[perm]

model=torch.nn.Sequential(
    torch.nn.Conv2d(3,32,3,padding=1), torch.nn.ReLU(),torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(32,64,3,padding=1), torch.nn.ReLU(),torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(64,128,3,padding=1), torch.nn.ReLU(),torch.nn.MaxPool2d(2),
    torch.nn.Flatten(),
    torch.nn.Linear(8192,256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(256,n_species),
)
model=model.to(device)

def scores_of(x):
    model.eval()
    out=[]
    with torch.no_grad():
        for i in range(0,len(x),500):
            out.append(model(x[i:i+500]))
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
        loss=loss_fn(model(xb),soft[yb])
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(epoch,accuracy(train_x[:2000],train_y[:2000]),accuracy(val_x,val_y),accuracy(test_x,test_y))

for name,x,t in [('train',train_x[:2000],train_y[:2000]),('val',val_x,val_y),('test',test_x,test_y)]:
    scores=scores_of(x)
    top1=(scores.argmax(dim=1)==t).float().mean().item()
    top5=(scores.topk(5,dim=1).indices==t[:,None]).any(dim=1).float().mean().item()
    print(name,'top1',top1,'top5',top5)

for name,x,t in [('val',val_x,val_y),('test',test_x,test_y)]:
    guess=scores_of(x).argmax(dim=1)
    wrong=guess!=t
    print('sev',name,'wrong',wrong.sum().item(),end=' ')
    for lv in sev_levels:
        same=species_group[lv][guess]==species_group[lv][t]
        print(lv,(wrong&same).sum().item(),end=' ')
    print()
