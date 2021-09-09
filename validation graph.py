import numpy as np
import pandas as pd
from InitializationNN import selectinputweight,selectoutputweight,hidden_output_weight,input_hidden_weight
from InitializationNN import sigmoid
import matplotlib.pyplot as plt
#x=np.array([0.768180528,0.760480888,0.768956289,0.785432473,0.152090798,50.734999])


ih1 = pd.read_csv('Dataset/Intel/Epoch/input_hidden1.csv')
ho1 = pd.read_csv('Dataset/Intel/Epoch/hidden_output1.csv')
data=pd.read_csv('Dataset/Intel/validate.csv')

ih2 = pd.read_csv('Dataset/Intel/Epoch/input_hidden2.csv')
ho2 = pd.read_csv('Dataset/Intel/Epoch/hidden_output2.csv')

ih3 = pd.read_csv('Dataset/Intel/Epoch/input_hidden3.csv')
ho3= pd.read_csv('Dataset/Intel/Epoch/hidden_output3.csv')

ih10 = pd.read_csv('Dataset/Intel/Epoch/input_hidden10.csv')
ho10 = pd.read_csv('Dataset/Intel/Epoch/hidden_output10.csv')
#assign weights
weight_input_hidden1= input_hidden_weight(ih1)
weight_hidden_output1= hidden_output_weight(ho1)

weight_input_hidden2= input_hidden_weight(ih2)
weight_hidden_output2= hidden_output_weight(ho2)

weight_input_hidden3= input_hidden_weight(ih3)
weight_hidden_output3= hidden_output_weight(ho3)

weight_input_hidden10= input_hidden_weight(ih10)
weight_hidden_output10= hidden_output_weight(ho10)
def runNN(weight_input_hidden,weight_hidden_output,x):
    for j in range(10):
    #hidden layer
        z=np.dot(weight_input_hidden,x[0:5])
        for i in range(10):
            z[i]=sigmoid(z[i])
        #print('The nodes in hidden layer are :',z)

    #output layer
        y=np.dot(weight_hidden_output,z)

    return y

'''y=runNN(x)
error=abs(((x[5]-y)/x[5])*100)
print('Data:',x)
print('Predicted Value:',y)
print('Accuracy:',100-error)'''
predicted_output1=[]
predicted_output2=[]
predicted_output3=[]
predicted_output10=[]
expected_output=[]
# Get all data
x=np.array([0,0,0,0,0,0],dtype=float)
for i in range (30):
    for qr in range(6):
        x[qr] = data.iloc[i][qr]
    expected_output.append(x[5])
    predicted_output1.append(runNN(weight_input_hidden1,weight_hidden_output1,x))
    predicted_output2.append(runNN(weight_input_hidden2, weight_hidden_output2, x))
    predicted_output3.append(runNN(weight_input_hidden3, weight_hidden_output3, x))
    predicted_output10.append(runNN(weight_input_hidden10, weight_hidden_output10, x))
print(expected_output)
#print(predicted_output)

'''for m in range(30):
    mape.iloc[m]=predicted_output[m]
mape.to_csv('Dataset/Nvidia/MAPE.csv', index=False)'''

#plt.xlim(30,0)
#plt.xticks(color='w')
plt.plot(predicted_output1,label='Iteration 1',color='gray',linestyle=':')
plt.plot(predicted_output2,label='Iteration 2',color='brown',linestyle=':')
plt.plot(predicted_output3,label='Iteration 3',color='orange',linestyle=':')
plt.plot(predicted_output10,label='Iteration 10',color='red')
plt.plot(expected_output,label='Actual Output',color='blue')
plt.ylabel('Average Stock Price')
plt.xlabel('Days')
plt.legend()
plt.show()

