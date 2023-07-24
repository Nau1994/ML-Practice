import numpy as np 

# Compute every step manually

# Linear regression
# f = w * x 

# here : f = 2 * x

X=np.array([1,2,3,4],dtype=np.float32)
Y=np.array([2,4,6,8],dtype=np.float32)

w=0.0

# model output
def forward(x):
    return w*x

#Loss=MSE
#j=MSE=1/N * (x*w-y)**2
def loss(y,y_pred):
    return ((y_pred-y)**2).mean()


#dj/dw =1/N * (2*x*(x*w - y))
def gradient(x,y,y_pred):
    return np.mean(2*x*(y_pred-y))

print(f'Prediction before training: f(5) = {forward(5):.3f}')

  
#Training 
learning_rate=0.01
n_iterate=25

for epoch in range(n_iterate):
    #predict =forword pass
    y_predict=forward(X)
    
    #loss
    l=loss(Y,y_predict)
    
    #calculate gradient
    dw=gradient(X,Y,y_predict)
    
    #update the weight
    w-=learning_rate*dw
    print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
     
print(f'Prediction after training: f(5) = {forward(5):.3f}')