import numpy as np
import torch

d=np.load('cifar10.npz')
x=torch.tensor(d['train_images']).float()/255
print(x.shape)
x=x.permute(0,3,1,2)
print(x.shape,x.dtype)
y=torch.tensor(d['train_labels']).flatten().long()
print(y[::5000])
torch.manual_seed(42)
order=torch.randperm(50000)
x=x[order]
y=y[order]
print(y[::5000])
train_x,train_y=x[:45000],y[:45000]
val_x,val_y=x[45000:],y[45000:]
print(train_x.shape)
print(val_x.shape)
block_1=torch.nn.Sequential(
    torch.nn.Conv2d(3,32,3,padding=1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(32,32,3),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
)
print(block_1(train_x[:1]).shape)

block_2=torch.nn.Sequential(
    torch.nn.Conv2d(32,64,3,padding=1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(64,64,3),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    
)
print(block_2(block_1(train_x[:1])).shape)

block_3=torch.nn.Sequential(
    torch.nn.Conv2d(64,64,3, padding=1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(64,64,3),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),

)
print(block_3(block_2(block_1(train_x[:1]))).shape)

model=torch.nn.Sequential(
    block_1,
    block_2,
    block_3,
    torch.nn.Flatten(),
    torch.nn.Linear(256,512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(512,10),


)
loss_fn=torch.nn.CrossEntropyLoss()
opt=torch.optim.NAdam(model.parameters(),lr=0.001)
for epoch in range(10):
    for i in range(0,45000,32):
        xb=train_x[i:i+32]
        yb=train_y[i:i+32]
        pred=model(xb)
        loss=loss_fn(pred,yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(epoch,loss.item())
model.eval()
with torch.no_grad():
    pred=model(val_x)
    acc=(pred.argmax(dim=1)==val_y).float().mean()
print(acc.item()) 
test_x=torch.tensor(d['test_images']).permute(0,3,1,2).float()/255
test_y=torch.tensor(d['test_labels']).flatten().long()
with torch.no_grad():
    pred=model(test_x)
    test_acc=(pred.argmax(dim=1)==test_y).float().mean()
print('test',test_acc.item())