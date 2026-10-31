import sys
import numpy as np
import torch
npz=sys.argv[1]
seed=int(sys.argv[2])
arch=sys.argv[3]
epochs=int(sys.argv[4])
torch.manual_seed(seed)

level_ids={'kingdom':1,'phylum':2,'class':3,'order':4,'family':5,'genus':6}

d=np.load(npz)
y=torch.tensor(d['y']).long()
names=[]
for n in d['names']:
    names.append(str(n))
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

device='mps'
if len(sys.argv)>5 and sys.argv[5]=='cpu':
    device='cpu'
train_x,train_y=train_x.to(device),train_y.to(device)
val_x,val_y=val_x.to(device),val_y.to(device)
test_x,test_y=test_x.to(device),test_y.to(device)
for lv in species_group:
    species_group[lv]=species_group[lv].to(device)

group_matrix={}
for lv in sev_levels:
    n_groups=int(species_group[lv].max().item())+1
    m=torch.zeros(n_species,n_groups,device=device)
    for s in range(n_species):
        m[s,species_group[lv][s]]=1
    group_matrix[lv]=m

perm=torch.randperm(n_train,device=device)
train_x=train_x[perm]
train_y=train_y[perm]

class ResBlock(torch.nn.Module):
    def __init__(self,c_in,c_out,stride):
        super().__init__()
        self.conv1=torch.nn.Conv2d(c_in,c_out,3,stride=stride,padding=1,bias=False)
        self.bn1=torch.nn.BatchNorm2d(c_out)
        self.conv2=torch.nn.Conv2d(c_out,c_out,3,padding=1,bias=False)
        self.bn2=torch.nn.BatchNorm2d(c_out)
        self.relu=torch.nn.ReLU()
        self.skip=torch.nn.Identity()
        if stride!=1 or c_in!=c_out:
            self.skip=torch.nn.Sequential(
                torch.nn.Conv2d(c_in,c_out,1,stride=stride,bias=False),
                torch.nn.BatchNorm2d(c_out),
            )
    def forward(self,x):
        out=self.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        return self.relu(out+self.skip(x))

if arch=='cnn':
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
if arch=='lenet':
    model=torch.nn.Sequential(
        torch.nn.Conv2d(3,6,5), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(6,16,5), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(16*13*13,120), torch.nn.ReLU(),
        torch.nn.Linear(120,84), torch.nn.ReLU(),
        torch.nn.Linear(84,n_species),
    )
if arch=='alexnet':
    model=torch.nn.Sequential(
        torch.nn.Conv2d(3,64,5,padding=2), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(64,192,3,padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(192,256,3,padding=1), torch.nn.ReLU(),
        torch.nn.Conv2d(256,256,3,padding=1), torch.nn.ReLU(),
        torch.nn.Conv2d(256,192,3,padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(192*8*8,512), torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512,512), torch.nn.ReLU(),
        torch.nn.Linear(512,n_species),
    )
if arch=='resnet':
    model=torch.nn.Sequential(
        torch.nn.Conv2d(3,32,3,padding=1,bias=False), torch.nn.BatchNorm2d(32), torch.nn.ReLU(),
        ResBlock(32,32,1),
        ResBlock(32,64,2),
        ResBlock(64,128,2),
        ResBlock(128,256,2),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(256,n_species),
    )
if arch=='resnet18':
    model=torch.nn.Sequential(
        torch.nn.Conv2d(3,64,3,stride=2,padding=1,bias=False), torch.nn.BatchNorm2d(64), torch.nn.ReLU(),
        ResBlock(64,64,1), ResBlock(64,64,1),
        ResBlock(64,128,2), ResBlock(128,128,1),
        ResBlock(128,256,2), ResBlock(256,256,1),
        ResBlock(256,512,2), ResBlock(512,512,1),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512,n_species),
    )
if arch=='resnet18_pre':
    import torchvision
    base=torchvision.models.resnet18(weights='IMAGENET1K_V1')
    base.fc=torch.nn.Linear(512,n_species)
    model=torch.nn.Sequential(torch.nn.Upsample(scale_factor=2,mode='bilinear'),base)
model=model.to(device)
print(arch,'parameters',sum(p.numel() for p in model.parameters()))

def scores_of(x):
    model.eval()
    out=[]
    with torch.no_grad():
        for i in range(0,len(x),500):
            out.append(model(x[i:i+500]))
    return torch.cat(out)

def accuracy(x,t):
    return (scores_of(x).argmax(dim=1)==t).float().mean().item()

def level_accuracy(x,t):
    probs=torch.softmax(scores_of(x),dim=1)
    out=[]
    for lv in sev_levels:
        group_probs=probs@group_matrix[lv]
        right=(group_probs.argmax(dim=1)==species_group[lv][t]).float().mean().item()
        out.append(right)
    return out

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
    print(epoch,accuracy(train_x[:2000],train_y[:2000]),accuracy(val_x,val_y),accuracy(test_x,test_y))
    la=level_accuracy(val_x,val_y)
    print('lvl',epoch,'class',la[0],'order',la[1],'family',la[2])

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
