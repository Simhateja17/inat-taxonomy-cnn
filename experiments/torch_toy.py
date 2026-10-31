import torch
x=torch.tensor(([1.0],[2.0],[3.0],[4.0]))
y=3*x

model=torch.nn.Linear(1,1)
loss_fn=torch.nn.MSELoss()
opt=torch.optim.SGD(model.parameters(),lr=0.1)

for epoch in range(5):
    pred=model(x)
    loss=loss_fn(pred,y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(epoch, loss.item())
    print(model.weight.item())