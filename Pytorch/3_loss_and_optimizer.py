
# 1) Design model (input, output, forward pass with different layers)
# 2) Construct loss and optimizer
# 3) Training loop
#       - Forward = compute prediction and loss
#       - Backward = compute gradients
#       - Update weightsimport torch

import torch
import torch.nn as nn

# Linear regression
# f = w * x 

# here : f = 2 * x

X=torch.tensor([1,2,3,4],dtype=torch.float32)
Y=torch.tensor([2,4,6,8],dtype=torch.float32)

w=torch.tensor(0.0,dtype=torch.float32,requires_grad=True)

# model output
def forward(x):
    return w*x

# 2) Define loss and optimizer
learning_rate = 0.01

# callable function
loss =nn.MSELoss()

#optimizer
optimizer=torch.optim.SGD([w],lr=learning_rate)

print(f'Prediction before training: f(5) = {forward(5):.3f}')

  
#Training 
n_iterate=100

for epoch in range(n_iterate):
    #predict =forword pass
    y_predict=forward(X)
    
    #loss
    l=loss(Y,y_predict)
    
    #calculate gradient
    l.backward()
    
    #update the weight
    optimizer.step()
    
     # zero the gradients after updating
    optimizer.zero_grad()
    
    
    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w.item():.3f}, loss = {l.item():.8f}')
     
print(f'Prediction after training: f(5) = {forward(5):.3f}')