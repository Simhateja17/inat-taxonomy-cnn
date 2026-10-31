import sys
import numpy as np
import torch
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
print(train_x.shape,val_x.shape,test_x.shape)
features=torch.nn.Sequential(
    torch.nn.Conv2d(3,32,3,padding=1), torch.nn.ReLU(),torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(32,64,3,padding=1), torch.nn.ReLU(),torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(64,128,3,padding=1), torch.nn.ReLU(),torch.nn.MaxPool2d(2),
)
model=torch.nn.Sequential(
    features,
    torch.nn.Flatten(),
    torch.nn.Linear(8192,256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(256,200),
)

print(model(train_x[:1]).shape)
print(sum(p.numel() for p in model.parameters()))
device='mps'
model=model.to(device)
train_x,train_y=train_x.to(device),train_y.to(device)
val_x,val_y=val_x.to(device),val_y.to(device)
test_x,test_y=test_x.to(device),test_y.to(device)
torch.manual_seed(seed)
perm=torch.randperm(6000,device=device)
train_x=train_x[perm]
train_y=train_y[perm]
print(train_y[:10])
loss_fn=torch.nn.CrossEntropyLoss()
opt=torch.optim.NAdam(model.parameters(),lr=0.0003)
for epoch in range(20):
    model.train()
    for i in range(0,6000,32):
        xb=train_x[i:i+32]
        if torch.rand(1).item()<0.5:
            xb=xb.flip(3)
        if torch.rand(1).item()<0.5:
            xb=xb.flip(2)
        yb=train_y[i:i+32]
        loss=loss_fn(model(xb),yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        train_acc=(model(train_x[:2000]).argmax(dim=1)==train_y[:2000]).float().mean()
        pred=model(val_x)
        val_acc=(pred.argmax(dim=1)==val_y).float().mean()
        test_acc=(model(test_x).argmax(dim=1)==test_y).float().mean()
    print(epoch,val_acc.item(),train_acc.item(),test_acc.item())


model.eval()
with torch.no_grad():
    for name,x,t in [('train',train_x[:2000],train_y[:2000]),('val',val_x,val_y),('test',test_x,test_y)]:
        scores=model(x)
        top1=(scores.argmax(dim=1)==t).float().mean().item()
        top5=(scores.topk(5,dim=1).indices==t[:,None]).any(dim=1).float().mean().item()
        print(name,'top1',top1,'top5',top5)
orders=[str(n).split('_')[4] for n in d['names']]
order_names=sorted(set(orders))
order_of_species=torch.tensor([order_names.index(o) for o in orders],device=device)
with torch.no_grad():
    guess=model(test_x).argmax(dim=1)
wrong=guess!=test_y
same_order=order_of_species[guess]==order_of_species[test_y]
print('wrong guesses',wrong.sum().item())
print('wrong but right order',(wrong&same_order).sum().item())

    