import numpy as np
import pandas as pd
from InitializationNN import input_hidden_weight,hidden_output_weight
from InitializationNN import sigmoid,sigmoid_p


ih = pd.read_csv('Dataset/Intel/Epoch/input_hidden9.csv')
ho = pd.read_csv('Dataset/Intel/Epoch/hidden_output9.csv')
data=pd.read_csv('Dataset/Intel/stocktotest.csv')
#initialize weight
weight_input_hidden= input_hidden_weight(ih)
weight_hidden_output= hidden_output_weight(ho)
x=np.array([0,0,0,0,0,0],dtype=float)
for xcount in range (1000):
    for qr in range(6):
        x[qr]=data.iloc[xcount][qr]
    print('The inputs are:', x)
    #for j in range(10):
    #hidden layer
    z=np.dot(weight_input_hidden,x[0:5])
    for i in range(10):
        z[i]=sigmoid(z[i])
    #print('The nodes in hidden layer are :',z)

#output layer
    y=np.dot(weight_hidden_output,z)

#print('The output is:',y)

#backporpagation section
    learning_rate=0.1

    error = ((x[5]-y)/x[5])*100

#backpropagation between hidden layer and outer layer
    for i in range(10):
        output_gradient=2*(y-x[5])*z[i]/1000
        weight_hidden_output[i]=weight_hidden_output[i]-learning_rate*output_gradient
    #between hidden layer and input layer
    for i in range(10):
        for k in range(5):
            hidden_gradient=2*(y-x[5])*sigmoid_p(weight_hidden_output[i])*weight_hidden_output[i]*x[k]/1000
            weight_input_hidden[i,k]=weight_input_hidden[i,k]-learning_rate*hidden_gradient
    #if j%1==0:
        #print('The weights between hidden and output are:', weight_hidden_output)
        #print('The weights between hidden and input are:',weight_input_hidden)
    y=np.dot(weight_hidden_output,z)
    print('Predicted output:',y)
    error = abs(((x[5]-y)/x[5])*100)
    print('Error:',error)

#update weight
for i in range(10):
       for j in range(5):
            ih.iloc[i][j]=weight_input_hidden[i][j]
for k in range(10):
    ho.iloc[k]=weight_hidden_output[k]
ih.to_csv('Dataset/Intel/Epoch/input_hidden10.csv',index=False)
ho.to_csv('Dataset/Intel/Epoch/hidden_output10.csv',index=False)
#print(ih)
#print(ho)