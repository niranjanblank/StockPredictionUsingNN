import pandas as pd
import matplotlib.pyplot as plt
ab=pd.read_csv('Dataset/AMD/data.csv')
a=ab.shape[0]
q=ab.iloc[1257,5]
x=[]
y=[]
j=30
for i in range(31):
    x.append(ab.iloc[a-1-j,5])
    j=j-1
print(x)
plt.plot(x)
plt.ylabel('Average Stock Price')
plt.show()



ih = pd.read_csv('Dataset/Tesla/3days/input_hidden.csv')
ho = pd.read_csv('Dataset/Tesla/3days/hidden_output.csv')
data=pd.read_csv('Dataset/Tesla/3days/stocktotest.csv')
def runNN(x):
   for j in range(10):
      # hidden layer
      z = np.dot(weight_input_hidden, x)
      for i in range(10):
         z[i] = sigmoid(z[i])
      # print('The nodes in hidden layer are :',z)

      # output layer
      y = np.dot(weight_hidden_output, z)

   return y

def normalize(x):
   for i in range(5):
      x[i]=(x[i]-minmax_value[1,i])/(minmax_value[0,i]-minmax_value[1,i])
   return x
def check():
   ## get values
   id_info= int(id)
   open_info=float(o_price.get())
   close_info = float(c_price.get())
   high_info = float(h_price.get())
   low_info = float(l_price.get())
   volume_info = float(volume.get())

   weightflag = id_info
   inputweight = selectinputweight(weightflag)
   outputweight = selectoutputweight(weightflag)
   minmaxvalue= selectminmax(weightflag)
   ih = pd.read_csv(inputweight)
   ho = pd.read_csv(outputweight)
   mm = pd.read_csv(minmaxvalue)
   # assign weights
   #screen1=Toplevel(screen)
   global weight_hidden_output
   global weight_input_hidden
   weight_input_hidden = input_hidden_weight(ih)
   weight_hidden_output = hidden_output_weight(ho)
   global minmax_value
   minmax_value = minmax(mm)
   x=[open_info,high_info,low_info,close_info,volume_info]
   new_x=normalize(x)
   predicted=runNN(new_x)
   print(predicted)
   printans.set(predicted)
   #Label(screen, text="Predicted Value", fg="green", font=("Calibri", 15)).pack()
   #Label(screen, text=predicted).pack()
