from tkinter import *
import pandas as pd
import numpy as np
from InitializationNN import selectinputweight,selectoutputweight,hidden_output_weight,input_hidden_weight,minmax,selectminmax
from InitializationNN import sigmoid

global correctanswer
def change(*args):
   global id
   print("running change")
   if var.get() == "Amazon":
      id = 1
   elif var.get() == "AMD":
      id = 2
   elif var.get()=="Apple":
      id = 3
   elif var.get()=="Cisco":
      id=4
   elif var.get()=="Facebook":
      id=5
   elif var.get()=="Google":
      id=6
   elif var.get()=="Intel":
      id= 7
   elif var.get()=="Microsoft":
      id=8
   elif var.get()=="Nvidia":
      id=9
   else:
      id=10
   print(id)
   '''1.
   Amazon\n2.AMD\n3.Apple\n4.Cisco\n5.Facebook\n6.Google\n7.Intel\n8.Microsoft\n9.Nvidia\n10.Tesla'''
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


def predict_screen():
   global screen
   global c_id
   global o_price
   global c_price
   global h_price
   global l_price
   global volume
   global toprint
   global var
   global answer
   global printans

   screen = Tk()
   screen.geometry("720x700")
   screen.title("Stock Prediction")
   Label(text="Stock Prediction", bg="grey", width=300,height=2,font = ("Calibri",15)).pack()
   c_id=StringVar()
   o_price=StringVar()
   c_price=StringVar()
   h_price=StringVar()
   l_price=StringVar()
   volume=StringVar()
   printans=StringVar()

   Label(text="Company IDs:\n1. Amazon\n2. AMD\n3. Apple\n4. Cisco\n5. Facebook\n6. Google\n7. Intel\n8. Microsoft\n9. Nvidia\n10. Tesla",
         bg="grey", width=20, height=720,font = ("Calibri",13)).pack(side="left")
   Label( text="").pack()
 #  Label( text="Company ID").pack()
  # id_entry = Entry( text=c_id)
  # id_entry.pack()
   OPTIONS = [
   "Amazon",
   "AMD",
   "Apple",
      "Cisco","Facebook","Google","Intel","Microsoft","Nvidia","Tesla"
   ]

   var = StringVar(screen)
   var.set(OPTIONS[0])
   var.trace("w", change)

   dropDownMenu = OptionMenu(screen, var, OPTIONS[0], OPTIONS[1], OPTIONS[2],OPTIONS[3],OPTIONS[4],OPTIONS[5],OPTIONS[6],OPTIONS[7],OPTIONS[8],OPTIONS[9])
   dropDownMenu.config(width=40)
   dropDownMenu.pack()
   Label(text="").pack()
   Label(text="Opening Price").pack()
   open_entry = Entry( text=o_price)
   open_entry.pack()
   Label(text="").pack()
   Label(text="Closing Price").pack()
   close_entry = Entry(text=c_price)
   close_entry.pack()
   Label(text="").pack()
   Label(text="Highest Price").pack()
   high_entry = Entry(text=h_price)
   high_entry.pack()
   Label(text="").pack()
   Label(text="Lowest Price").pack()
   low_entry = Entry(text=l_price)
   low_entry.pack()
   Label(text="").pack()
   Label(text="Volume").pack()
   volume_entry = Entry(text=volume)
   volume_entry.pack()
   Button(text="Predict", width=10, height=1, command=check).pack()
   Label(text="").pack()
   Label(text="").pack()
   Label(screen, text="Predicted Value", fg="green", font=("Calibri", 15)).pack()
   Label(textvariable=printans,width=30,bg='white').pack()
   screen.mainloop()

predict_screen()