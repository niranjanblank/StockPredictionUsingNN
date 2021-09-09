from tkinter import *
import pandas as pd
import numpy as np
from InitializationNN import selectinputweight,selectoutputweight,hidden_output_weight,input_hidden_weight,minmax,selectminmax,selectinputweight_three,selectoutputweight_three
from InitializationNN import sigmoid
def runNNforview(x,a,sel_day):
    global weight_hidden_output
    global weight_input_hidden
    weightflag = a
    if sel_day==1:
        inputweight = selectinputweight(weightflag)
        outputweight = selectoutputweight(weightflag)
    if sel_day==3:
        inputweight = selectinputweight_three(weightflag)
        outputweight = selectoutputweight_three(weightflag)
    ih = pd.read_csv(inputweight)
    print(inputweight)
    print(outputweight)
    ho = pd.read_csv(outputweight)
    weight_input_hidden = input_hidden_weight(ih)
    weight_hidden_output = hidden_output_weight(ho)
    for j in range(10):
        # hidden layer
        z = np.dot(weight_input_hidden, x)
        for i in range(10):
            z[i] = sigmoid(z[i])
        # print('The nodes in hidden layer are :',z)

        # output layer
        y = np.dot(weight_hidden_output, z)

    return y


def normalizeforview(x,a):
    global mm
    global minmax_value
    minmaxvalue = selectminmax(a)
    mm = pd.read_csv(minmaxvalue)
    minmax_value = minmax(mm)
    for i in range(5):
      x[i]=(x[i]-minmax_value[1,i])/(minmax_value[0,i]-minmax_value[1,i])
    return x

def getdata():
    data=pd.read_csv("CurrentValues/current.csv")
    x = np.array([(0, 0, 0, 0, 0),
                  (0, 0, 0, 0, 0),
                  (0, 0, 0, 0, 0),
                  (0, 0, 0, 0, 0),
                  (0, 0, 0, 0, 0),
                  (0, 0, 0, 0, 0),
                  (0, 0, 0, 0, 0),
                  (0, 0, 0, 0, 0),
                  (0, 0, 0, 0, 0),
                  (0, 0, 0, 0, 0)], dtype=float)
    for i in range(10):
        for j in range(5):
            x[i][j] = data.iloc[i][j]
    return x

def getnormalizeddata(x):
    dat=[]
    for i in range(10):
        dat.append(normalizeforview(x[i],(i+1)))
    return(dat)

def getpredicteddata():
    y=getdata()
    #one day prediction
    ans_one=getnormalizeddata(y)

    for i in range(10):
        stock_one.append(runNNforview(ans_one[i],(i+1),1))

#for i in range(10):
    #print(stock_one[i])
#three day prediction
    x=getdata()
    ans_three=getnormalizeddata(x)

    for i in range(10):

        stock_three.append(runNNforview(ans_three[i],(i+1),3))

    for i in range(10):
        print(stock_three[i])
def value_table():
   value_window = Tk()
   value_window.geometry("1480x480")
   value_window.title("data")
   global stock_one
   stock_one=[]
   global stock_three
   stock_three=[]
   getpredicteddata()
   l1 = Label(value_window, text=" Stock Prediction  ",height=3,font=("Calibri",13))
   l1.grid(row=0, column=4)
   l1 = Label(value_window, text=" S.N.  ",bg="grey",fg="white",relief="raised",height=2,width=7)
   l1.grid(row=1, column=0)
   l1 = Label(value_window, text=" Traded Companies ",bg="grey",fg="white",relief="raised",height=2,width=20)
   l1.grid(row=1, column=1)
   l1 = Label(value_window, text=" Open price ",bg="grey",fg="white",width=20,relief="raised",height=2)
   l1.grid(row=1, column=2)
   l1 = Label(value_window, text=" Highest Price ",bg="grey",fg="white",width=20,relief="raised",height=2)
   l1.grid(row=1, column=3)
   l1 = Label(value_window, text=" Lowest price ",bg="grey",fg="white",width=20,relief="raised",height=2)
   l1.grid(row=1, column=4)
   l1 = Label(value_window, text=" Close Price ",bg="grey",fg="white",width=20,relief="raised",height=2)
   l1.grid(row=1, column=5)
   l1 = Label(value_window, text="  Volume  ",bg="grey",fg="white",width=20,relief="raised",height=2)
   l1.grid(row=1, column=6)
   l1 = Label(value_window, text="  Predicted Average Stock\n(1 Day)",bg="grey",fg="white",width=25,relief="raised",height=2)
   l1.grid(row=1, column=7)
   l1 = Label(value_window, text="  Predicted Average Stock\n(3 Days)", bg="grey", fg="white", width=25, relief="raised",
              height=2)
   l1.grid(row=1, column=8)
   data = pd.read_csv("CurrentValues/current.csv")
   company = ['Amazon', 'AMD', 'Apple', 'Cisco', 'Facebook', 'Google', 'Intel', 'Microsoft', 'Nvidia', 'Tesla']
   for i in range(10):
       l1 = Label(value_window, text=(i+1),pady=5,relief="ridge",width=7)
       l1.grid(row=i+4, column=0)
       l1 = Label(value_window, text=company[i],relief="ridge",width=20,pady=5)
       l1.grid(row=i+4, column=1)
       l1 = Label(value_window, text=round(data.iloc[i][0],2),relief="ridge",width=20,pady=5)
       l1.grid(row=i+4, column=2)
       l1 = Label(value_window, text=round(data.iloc[i][1],2),relief="ridge",width=20,pady=5)
       l1.grid(row=i+4, column=3)
       l1 = Label(value_window, text=round(data.iloc[i][2],2),relief="ridge",width=20,pady=5)
       l1.grid(row=i+4, column=4)
       l1 = Label(value_window, text=round(data.iloc[i][3],2),relief="ridge",width=20,pady=5)
       l1.grid(row=i+4, column=5)
       l1 = Label(value_window, text=int(data.iloc[i][4]),relief="ridge",width=20,pady=5)
       l1.grid(row=i+4, column=6)
       l1 = Label(value_window, text=round(stock_one[i],2),pady=5,relief="ridge",width=25)
       l1.grid(row=i+4, column=7)
       l1 = Label(value_window, text=round(stock_three[i],2), pady=5, relief="ridge", width=25)
       l1.grid(row=i + 4, column=8)
   value_window.mainloop()

value_table()