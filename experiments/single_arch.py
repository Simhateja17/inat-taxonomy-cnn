import sys
import numpy as np
import torch
npz=sys.argv[1]
seed=int(sys.argv[2])
level=sys.argv[3]
epochs=int(sys.argv[4])
arch=sys.argv[5]
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
print(arch,level,n_groups,'groups',train_x.shape,val_x.shape,test_x.shape)

counts=torch.bincount(test_y,minlength=n_groups)
print('chance',1/n_groups,'biggest group share on test',counts.max().item()/len(test_y))

device='mps'
if len(sys.argv)>6 and sys.argv[6]=='cpu':
    device='cpu'
train_x,train_y=train_x.to(device),train_y.to(device)
val_x,val_y=val_x.to(device),val_y.to(device)
test_x,test_y=test_x.to(device),test_y.to(device)

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

if arch=='lenet':
    model=torch.nn.Sequential(
        torch.nn.Conv2d(3,6,5), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(6,16,5), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(16*5*5,120), torch.nn.ReLU(),
        torch.nn.Linear(120,84), torch.nn.ReLU(),
        torch.nn.Linear(84,n_groups),
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
        torch.nn.Linear(192*4*4,512), torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512,512), torch.nn.ReLU(),
        torch.nn.Linear(512,n_groups),
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
        torch.nn.Linear(256,n_groups),
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
        torch.nn.Linear(512,n_groups),
    )
if arch=='resnet18_pre':
    import torchvision
    base=torchvision.models.resnet18(weights='IMAGENET1K_V1')
    base.fc=torch.nn.Linear(512,n_groups)
    model=torch.nn.Sequential(torch.nn.Upsample(scale_factor=2,mode='bilinear'),base)
model=model.to(device)
print('parameters',sum(p.numel() for p in model.parameters()))

def scores_of(x):
    model.eval()
    out=[]
    with torch.no_grad():
        for i in range(0,len(x),250):
            out.append(model(x[i:i+250]))
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
        if (i//32)%100==0:
            loss.item()
    print(epoch,accuracy(train_x[:5000],train_y[:5000]),accuracy(val_x,val_y),accuracy(test_x,test_y),flush=True)

k=min(5,n_groups)
for name,x,t in [('train',train_x[:5000],train_y[:5000]),('val',val_x,val_y),('test',test_x,test_y)]:
    scores=scores_of(x)
    top1=(scores.argmax(dim=1)==t).float().mean().item()
    top5=(scores.topk(k,dim=1).indices==t[:,None]).any(dim=1).float().mean().item()
    print(name,'top1',top1,'top5',top5)
