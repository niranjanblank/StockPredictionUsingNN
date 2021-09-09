import numpy as np
import pandas as pd
from InitializationNN import selectinputweight,selectoutputweight,hidden_output_weight,input_hidden_weight
from InitializationNN import sigmoid
x=np.array([0.254204698,0.23879134,0.254930772,0.25806985,0.592964656,217.199997])
weightflag=7 ##6 initializes googles stockweights
inputweight=selectinputweight(weightflag)
outputweight=selectoutputweight(weightflag)

ih = pd.read_csv('Dataset/Tesla/3days/input_hidden.csv')
ho = pd.read_csv('Dataset/Tesla/3days/hidden_output.csv')
data=pd.read_csv('Dataset/Tesla/3days/stocktotest.csv')
#assign weights
weight_input_hidden= input_hidden_weight(ih)
weight_hidden_output= hidden_output_weight(ho)
def runNN(x):
    for j in range(10):
    #hidden layer
        z=np.dot(weight_input_hidden,x[0:5])
        for i in range(10):
            z[i]=sigmoid(z[i])
        #print('The nodes in hidden layer are :',z)

    #output layer
        y=np.dot(weight_hidden_output,z)

    return y

y=runNN(x)
error=abs(((x[5]-y)/x[5])*100)
print('Data:',x)
print('Predicted Value:',y)
print('Accuracy:',100-error)