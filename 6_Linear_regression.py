
# 1) Design model (input, output, forward pass with different layers)
# 2) Construct loss and optimizer
# 3) Training loop
#       - Forward = compute prediction and loss
#       - Backward = compute gradients
#       - Update weightsimport torch

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


#prepare data
x_numpy,y_numpy=datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=4)

#cast to float tensor
x=torch.from_numpy(x_numpy.astype(np.float32))
y=torch.from_numpy(y_numpy.astype(np.float32))

y=y.view(y.shape[0],1)

n_samples,n_features=x.shape

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
criterion  =nn.MSELoss()

#optimizer
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
 
#Training 
n_iterate=200

for epoch in range(n_iterate):
    # predict = forward pass with our model
    y_predict=model(x)
    
    #loss
    l=criterion (y,y_predict)
    
    #calculate gradient
    l.backward()
    
    #update the weight
    optimizer.step()
    
     # zero the gradients after updating
    optimizer.zero_grad()
    
    
    if epoch % 10 == 0:
        [w, b] = model.parameters() # unpack parameters
        print(f'epoch {epoch+1}: w = {w.item():.3f}, loss = {l.item():.8f}')
     

# Plot
predicted=model(x).detach().numpy()

plt.plot(x_numpy,y_numpy,'go')
plt.plot(x_numpy,predicted,'b')
plt.show()