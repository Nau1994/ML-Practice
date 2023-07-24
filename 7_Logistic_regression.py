
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#prepare data
bc=datasets.load_breast_cancer()
x_numpy,y_numpy=bc.data,bc.target

n_samples,n_features=x_numpy.shape

x_train,x_test,y_train,y_test=train_test_split(x_numpy,y_numpy,test_size=0.2,random_state=1234)

#scale
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#cast to float tensor
x_train=torch.from_numpy(x_train.astype(np.float32))
y_train=torch.from_numpy(y_train.astype(np.float32))
x_test=torch.from_numpy(x_test.astype(np.float32))
y_test=torch.from_numpy(y_test.astype(np.float32))

y_train=y_train.view(y_train.shape[0],1)
y_test=y_test.view(y_test.shape[0],1)

input_size = n_features
output_size = 1



#model implementation
class LogisticRegression(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(LogisticRegression,self).__init__()
        self.lin=nn.Linear(input_dim,output_dim)
    def forward(self,x):
        y_pred=torch.sigmoid(self.lin(x))
        return y_pred

# we can call this model with samples X
model=LogisticRegression(input_size,output_size);


# 2) Define loss and optimizer
learning_rate = 0.01

# callable function
criterion  =nn.BCELoss()

#optimizer
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
 
    
#Training 
n_iterate=200

for epoch in range(n_iterate):
    # predict = forward pass with our model
    y_predict=model(x_train)
    
    #loss
    l=criterion(y_predict,y_train)
    
    #calculate gradient
    l.backward()
    
    #update the weight
    optimizer.step()
    
     # zero the gradients after updating
    optimizer.zero_grad()
    
    
    if epoch % 10 == 0:
        # [w, b] = model.parameters() # unpack parameters
        print(f'epoch {epoch+1}, loss = {l.item():.4f}')
     
with torch.no_grad():
    y_predicted = model(x_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy: {acc.item():.4f}')
    
# Plot
predicted=model(x_test).detach().numpy()
# print(sc.inverse_transform(x_train.numpy())[:,1])
plt.plot(sc.inverse_transform(x_test.numpy())[:,0],y_test,'go')
plt.plot(sc.inverse_transform(x_test.numpy())[:,0],predicted,'bo')
plt.show()