import sys
import numpy as np
import torch
npz=sys.argv[1]
seed=int(sys.argv[2])
level_text=sys.argv[3]
epochs=int(sys.argv[4])
torch.manual_seed(seed)

level_ids={'kingdom':1,'phylum':2,'class':3,'order':4,'family':5,'genus':6}
levels=[]
if level_text!='none':
    levels=level_text.split(',')
shuffle=False
if len(sys.argv)>5 and sys.argv[5]=='shuffle':
    shuffle=True

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
    return torch.tensor(numbers),len(group_names)

sev_levels=['class','order','family']
species_group={}
group_count={}
for lv in sev_levels+levels:
    species_group[lv],group_count[lv]=group_of_species(lv)
    print(lv,group_count[lv],'groups')

device='mps'
if len(sys.argv)>6 and sys.argv[6]=='cpu':
    device='cpu'
train_x,train_y=train_x.to(device),train_y.to(device)
val_x,val_y=val_x.to(device),val_y.to(device)
test_x,test_y=test_x.to(device),test_y.to(device)
for lv in species_group:
    species_group[lv]=species_group[lv].to(device)

head_group={}
for lv in levels:
    if shuffle:
        g=torch.Generator()
        g.manual_seed(1000+seed)
        order=torch.randperm(n_species,generator=g).to(device)
        head_group[lv]=species_group[lv][order]
    else:
        head_group[lv]=species_group[lv]

perm=torch.randperm(n_train,device=device)
train_x=train_x[perm]
train_y=train_y[perm]

train_level=[]
for lv in levels:
    train_level.append(head_group[lv][train_y])

features=torch.nn.Sequential(
    torch.nn.Conv2d(3,32,3,padding=1), torch.nn.ReLU(),torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(32,64,3,padding=1), torch.nn.ReLU(),torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(64,128,3,padding=1), torch.nn.ReLU(),torch.nn.MaxPool2d(2),
)
body=torch.nn.Sequential(
    features,
    torch.nn.Flatten(),
    torch.nn.Linear(8192,256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
)
species_head=torch.nn.Linear(256,n_species)
level_heads=torch.nn.ModuleList()
for lv in levels:
    level_heads.append(torch.nn.Linear(256,group_count[lv]))
body=body.to(device)
species_head=species_head.to(device)
level_heads=level_heads.to(device)

def scores_of(x,head):
    body.eval()
    out=[]
    with torch.no_grad():
        for i in range(0,len(x),500):
            h=body(x[i:i+500])
            out.append(head(h))
    return torch.cat(out)

def accuracy(x,t):
    return (scores_of(x,species_head).argmax(dim=1)==t).float().mean().item()

loss_fn=torch.nn.CrossEntropyLoss()
params=list(body.parameters())+list(species_head.parameters())+list(level_heads.parameters())
opt=torch.optim.NAdam(params,lr=0.0003)
for epoch in range(epochs):
    body.train()
    for i in range(0,n_train,32):
        xb=train_x[i:i+32]
        yb=train_y[i:i+32]
        if torch.rand(1).item()<0.5:
            xb=xb.flip(3)
        if torch.rand(1).item()<0.5:
            xb=xb.flip(2)
        h=body(xb)
        loss=loss_fn(species_head(h),yb)
        for k in range(len(levels)):
            loss=loss+loss_fn(level_heads[k](h),train_level[k][i:i+32])
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(epoch,accuracy(train_x[:2000],train_y[:2000]),accuracy(val_x,val_y),accuracy(test_x,test_y))

for name,x,t in [('train',train_x[:2000],train_y[:2000]),('val',val_x,val_y),('test',test_x,test_y)]:
    scores=scores_of(x,species_head)
    top1=(scores.argmax(dim=1)==t).float().mean().item()
    top5=(scores.topk(5,dim=1).indices==t[:,None]).any(dim=1).float().mean().item()
    print(name,'top1',top1,'top5',top5)

for k in range(len(levels)):
    lv=levels[k]
    answer=head_group[lv][test_y]
    acc=(scores_of(test_x,level_heads[k]).argmax(dim=1)==answer).float().mean().item()
    print('acc_'+lv,acc)

for name,x,t in [('val',val_x,val_y),('test',test_x,test_y)]:
    guess=scores_of(x,species_head).argmax(dim=1)
    wrong=guess!=t
    print('sev',name,'wrong',wrong.sum().item(),end=' ')
    for lv in sev_levels:
        same=species_group[lv][guess]==species_group[lv][t]
        print(lv,(wrong&same).sum().item(),end=' ')
    print()
