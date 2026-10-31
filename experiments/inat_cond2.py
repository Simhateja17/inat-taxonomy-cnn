import numpy as np
import torch
import sys
seed=int(sys.argv[1])
torch.manual_seed(seed)
d=np.load('/Users/teja/inat_data/inat200_64.npz')
position=torch.arange(10000)%50
train=position<30
val=(position>=30)&(position<40)
test=position>=40
print(train.sum().item(),val.sum().item(),test.sum().item())
X=torch.tensor(d['X']).permute(0,3,1,2).contiguous().float()/255
X=(X-0.5)/0.25
y=torch.tensor(d['y']).long()
train_x,train_y=X[train],y[train]
val_x,val_y=X[val],y[val]
test_x,test_y=X[test],y[test]
orders=[]
for n in d['names']:
    orders.append(str(n).split('_')[4])
order_names=sorted(set(orders))
order_of_species_list=[]
for o in orders:
    order_of_species_list.append(order_names.index(o))
order_of_species=torch.tensor(order_of_species_list)
y_order=order_of_species[y]
train_order,val_order,test_order=y_order[train],y_order[val],y_order[test]
print(train_order.shape)

print(train_x.shape,val_x.shape,test_x.shape)
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
species_head=torch.nn.Linear(256,200)
order_head=torch.nn.Linear(256,20)
h=body(train_x[:1])
print(species_head(h).shape,order_head(h).shape)
device='mps'
body=body.to(device)
species_head=species_head.to(device)
order_head=order_head.to(device)
train_x,train_y=train_x.to(device),train_y.to(device)
val_x,val_y=val_x.to(device),val_y.to(device)
test_x,test_y=test_x.to(device),test_y.to(device)
train_order,val_order,test_order=train_order.to(device),val_order.to(device),test_order.to(device)
order_of_species=order_of_species.to(device)
torch.manual_seed(seed)
perm=torch.randperm(6000,device=device)
train_x=train_x[perm]
train_y=train_y[perm]
train_order=train_order[perm]
print(train_y[:10])
loss_fn=torch.nn.CrossEntropyLoss()
params=list(body.parameters())+list(species_head.parameters())+list(order_head.parameters())
opt=torch.optim.NAdam(params,lr=0.0003)
for epoch in range(20):
    body.train()
    for i in range(0,6000,32):
        xb=train_x[i:i+32]
        yb=train_y[i:i+32]
        ob=train_order[i:i+32]
        if torch.rand(1).item()<0.5:
            xb=xb.flip(3)
        if torch.rand(1).item()<0.5:
            xb=xb.flip(2)
        h=body(xb)
        loss=loss_fn(species_head(h),yb)+loss_fn(order_head(h),ob)
        opt.zero_grad()
        loss.backward()
        opt.step()
    body.eval()
    with torch.no_grad():
        train_acc=(species_head(body(train_x[:2000])).argmax(dim=1)==train_y[:2000]).float().mean()
        val_acc=(species_head(body(val_x)).argmax(dim=1)==val_y).float().mean()
        train_order_acc=(order_head(body(train_x[:2000])).argmax(dim=1)==train_order[:2000]).float().mean()
        val_order_acc=(order_head(body(val_x)).argmax(dim=1)==val_order).float().mean()
        test_acc=(species_head(body(test_x)).argmax(dim=1)==test_y).float().mean()
        print(epoch,train_acc.item(),val_order_acc.item(),val_acc.item(),test_acc.item())

with torch.no_grad():
    for name,x,t in [('train',train_x[:2000],train_y[:2000]),('val',val_x,val_y),('test',test_x,test_y)]:
        scores=species_head(body(x))
        top1=(scores.argmax(dim=1)==t).float().mean().item()
        top5=(scores.topk(5,dim=1).indices==t[:,None]).any(dim=1).float().mean().item()
        print(name,'top1',top1,'top5',top5)
    guess=species_head(body(test_x)).argmax(dim=1)
wrong=guess!=test_y
same_order=order_of_species[guess]==order_of_species[test_y]
print('wrong guesses',wrong.sum().item())
print('wrong but right order',(wrong&same_order).sum().item())
