
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

X=torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
Y=torch.tensor([[2],[4],[6],[8]],dtype=torch.float32)

n_samples,n_features=X.shape
print(f'#samples: {n_samples}, #features: {n_features}')

X_test=torch.tensor([5],dtype=torch.float32)

input_size = n_features
output_size = n_features



#model implementation
class LinearRegression(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(LinearRegression,self).__init__()
        self.lin=nn.Linear(input_dim,output_dim)
    def forward(self,x):
        return self.lin(x)

# we can call this model with samples X
model=LinearRegression(input_size,output_size);


# 2) Define loss and optimizer
learning_rate = 0.01

# callable function
loss =nn.MSELoss()

#optimizer
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')
  
#Training 
n_iterate=200

for epoch in range(n_iterate):
    # predict = forward pass with our model
    y_predict=model(X)
    
    #loss
    l=loss(Y,y_predict)
    
    #calculate gradient
    l.backward()
    
    #update the weight
    optimizer.step()
    
     # zero the gradients after updating
    optimizer.zero_grad()
    
    
    if epoch % 10 == 0:
        [w, b] = model.parameters() # unpack parameters
        print(f'epoch {epoch+1}: w = {w.item():.3f}, loss = {l.item():.8f}')
     
print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')