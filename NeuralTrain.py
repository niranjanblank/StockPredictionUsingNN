import numpy as np
import pandas as pd
from InitializationNN import input_hidden_weight
from InitializationNN import hidden_output_weight


def sigmoid(x):
    return (1/(1+np.exp(-x)))
def sigmoid_p(x):
    return sigmoid(x)* (1-sigmoid(x))
def relu(x):
    if x>0:
        return x
    else:
        return 0
def relu_p(x):
    if x>0:
        return(1)
    else:
        return(0)
ih = pd.read_csv('input_hidden.csv')
ho = pd.read_csv('hidden_output.csv')
#initialize weight
weight_input_hidden= input_hidden_weight()
weight_hidden_output= hidden_output_weight()
x=np.array([0.8717784878,0.8853889943,0.9141245299,0.9713742368,0.2590272503,331.475])
print('The inputs are:',x)
for j in range(10):
#hidden layer
    z=np.dot(weight_input_hidden,x[0:5])
    for i in range(10):
        z[i]=sigmoid(z[i])
    #print('The nodes in hidden layer are :',z)

#output layer
    y=np.dot(weight_hidden_output,z)
   # y=relu(y)
    #print('The output is:',y)

#backporpagation section
    learning_rate=0.1

    error = ((x[5]-y)/x[5])*100

#backpropagation between hidden layer and outer layer
    for i in range(10):
        output_gradient=2*(y-x[5])*z[i]/2
        weight_hidden_output[i]=weight_hidden_output[i]-learning_rate*output_gradient
#between hidden layer and input layer
    for i in range(10):
       for k in range(5):
            hidden_gradient=2*(y-x[5])*sigmoid_p(weight_hidden_output[i])*weight_hidden_output[i]*x[k]/2
            weight_input_hidden[i,k]=weight_input_hidden[i,k]-learning_rate*hidden_gradient
    if j%1==0:
        print('The weights between hidden and output are:', weight_hidden_output)
        print('The weights between hidden and input are:',weight_input_hidden)
y=np.dot(weight_hidden_output,z)
print('Predicted output:',y)
error = ((x[5]-y)/x[5])*100
print(error)


#update weight
for i in range(10):
        for j in range(5):
            ih.iloc[i][j]=weight_input_hidden[i][j]
for k in range(10):
    ho.iloc[k]=weight_hidden_output[k]
ih.to_csv('input_hidden.csv',index=False)
ho.to_csv('hidden_output.csv',index=False)
print(ih)
print(ho)