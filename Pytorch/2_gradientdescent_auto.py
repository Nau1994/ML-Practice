import torch

# Compute every step manually

# Linear regression
# f = w * x 

# here : f = 2 * x

X=torch.tensor([1,2,3,4],dtype=torch.float32)
Y=torch.tensor([2,4,6,8],dtype=torch.float32)

w=torch.tensor(0.0,dtype=torch.float32,requires_grad=True)

# model output
def forward(x):
    return w*x

#Loss=MSE
#j=MSE=1/N * (x*w-y)**2
def loss(y,y_pred):
    return ((y_pred-y)**2).mean()


#dj/dw =1/N * (2*x*(x*w - y))
#will be calculated by torch
# def gradient(x,y,y_pred):
#     return np.mean(2*x*(y_pred-y))

print(f'Prediction before training: f(5) = {forward(5):.3f}')

  
#Training 
learning_rate=0.01
n_iterate=100

for epoch in range(n_iterate):
    #predict =forword pass
    y_predict=forward(X)
    
    #loss
    l=loss(Y,y_predict)
    
    #calculate gradient
    l.backward()
    
    #update the weight
    with torch.no_grad():
        w-=learning_rate*w.grad
    
    w.grad.zero_()
    
    
    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w.item():.3f}, loss = {l.item():.8f}')
     
print(f'Prediction after training: f(5) = {forward(5):.3f}')