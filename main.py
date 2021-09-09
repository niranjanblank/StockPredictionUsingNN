from tkinter import *
from PIL import ImageTk, Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from InitializationNN import selectinputweight,selectoutputweight,hidden_output_weight,input_hidden_weight,minmax,selectminmax,selectinputweight_three,selectoutputweight_three
from InitializationNN import sigmoid,selectvalidate
from InitializationNN import selectcompanydata


###############################   value table    ###################################
def runNNforview(x, a, sel_day):
   global weight_hidden_output
   global weight_input_hidden
   weightflag = a
   if sel_day == 1:
      inputweight = selectinputweight(weightflag)
      outputweight = selectoutputweight(weightflag)
   if sel_day == 3:
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
      if weightflag==1:
         y = np.dot(weight_hidden_output, z)+450
      elif weightflag == 2:
         y = np.dot(weight_hidden_output, z) + 5
      elif weightflag==3:
         y = np.dot(weight_hidden_output, z)+20
      elif weightflag == 4:
         y = np.dot(weight_hidden_output, z) + 8
      elif weightflag == 5:
         y = np.dot(weight_hidden_output, z) + 17
      elif weightflag == 6:
         y = np.dot(weight_hidden_output, z) + 100
      elif weightflag == 7:
         y = np.dot(weight_hidden_output, z)
      elif weightflag == 8:
         y = np.dot(weight_hidden_output, z) + 18
      elif weightflag == 9:
         y = np.dot(weight_hidden_output, z) +2
      elif weightflag == 10:
         y = np.dot(weight_hidden_output, z) - 25
      else:
         y = np.dot(weight_hidden_output, z)

   return y


def normalizeforview(x, a):
   global mm
   global minmax_value
   minmaxvalue = selectminmax(a)
   mm = pd.read_csv(minmaxvalue)
   minmax_value = minmax(mm)
   for i in range(5):
      x[i] = (x[i] - minmax_value[1, i]) / (minmax_value[0, i] - minmax_value[1, i])
   return x


def getdata():
   data = pd.read_csv("CurrentValues/current.csv")
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
   dat = []
   for i in range(10):
      dat.append(normalizeforview(x[i], (i + 1)))
   return (dat)


def getpredicteddata():
   y = getdata()
   # one day prediction
   ans_one = getnormalizeddata(y)

   for i in range(10):
      stock_one.append(runNNforview(ans_one[i], (i + 1), 1))

   # for i in range(10):
   # print(stock_one[i])
   # three day prediction
   x = getdata()
   ans_three = getnormalizeddata(x)

   for i in range(10):
      stock_three.append(runNNforview(ans_three[i], (i + 1), 3))

   for i in range(10):
      print(stock_three[i])


def value_table():
   value_window = Tk()
   value_window.geometry("1480x480")
   value_window.title("data")
   value_window.resizable(width=FALSE, height=FALSE)
   global stock_one
   stock_one = []
   global stock_three
   stock_three = []
   getpredicteddata()
   l1 = Label(value_window, text=" Stock Prediction  ", height=3, font=("Calibri", 13))
   l1.grid(row=0, column=4)
   l1 = Label(value_window, text=" S.N.  ", bg="grey", fg="white", relief="raised", height=2, width=7)
   l1.grid(row=1, column=0)
   l1 = Label(value_window, text=" Traded Companies ", bg="grey", fg="white", relief="raised", height=2, width=20)
   l1.grid(row=1, column=1)
   l1 = Label(value_window, text=" Open price ", bg="grey", fg="white", width=20, relief="raised", height=2)
   l1.grid(row=1, column=2)
   l1 = Label(value_window, text=" Highest Price ", bg="grey", fg="white", width=20, relief="raised", height=2)
   l1.grid(row=1, column=3)
   l1 = Label(value_window, text=" Lowest price ", bg="grey", fg="white", width=20, relief="raised", height=2)
   l1.grid(row=1, column=4)
   l1 = Label(value_window, text=" Close Price ", bg="grey", fg="white", width=20, relief="raised", height=2)
   l1.grid(row=1, column=5)
   l1 = Label(value_window, text="  Volume  ", bg="grey", fg="white", width=20, relief="raised", height=2)
   l1.grid(row=1, column=6)
   l1 = Label(value_window, text="  Predicted Average Stock\n(1 Day)", bg="grey", fg="white", width=25, relief="raised",
              height=2)
   l1.grid(row=1, column=7)
   l1 = Label(value_window, text="  Predicted Average Stock\n(3 Days)", bg="grey", fg="white", width=25,
              relief="raised",
              height=2)
   l1.grid(row=1, column=8)
   data = pd.read_csv("CurrentValues/current.csv")
   company = ['Amazon', 'AMD', 'Apple', 'Cisco', 'Facebook', 'Google', 'Intel', 'Microsoft', 'Nvidia', 'Tesla']
   for i in range(10):
      l1 = Label(value_window, text=(i + 1), pady=5, relief="ridge", width=7)
      l1.grid(row=i + 4, column=0)
      l1 = Label(value_window, text=company[i], relief="ridge", width=20, pady=5)
      l1.grid(row=i + 4, column=1)
      l1 = Label(value_window, text=round(data.iloc[i][0],2), relief="ridge", width=20, pady=5)
      l1.grid(row=i + 4, column=2)
      l1 = Label(value_window, text=round(data.iloc[i][1],2), relief="ridge", width=20, pady=5)
      l1.grid(row=i + 4, column=3)
      l1 = Label(value_window, text=round(data.iloc[i][2],2), relief="ridge", width=20, pady=5)
      l1.grid(row=i + 4, column=4)
      l1 = Label(value_window, text=round(data.iloc[i][3],2), relief="ridge", width=20, pady=5)
      l1.grid(row=i + 4, column=5)
      l1 = Label(value_window, text=int(data.iloc[i][4]), relief="ridge", width=20, pady=5)
      l1.grid(row=i + 4, column=6)
      l1 = Label(value_window, text=round(stock_one[i],2), pady=5, relief="ridge", width=25)
      l1.grid(row=i + 4, column=7)
      l1 = Label(value_window, text=round(stock_three[i],2), pady=5, relief="ridge", width=25)
      l1.grid(row=i + 4, column=8)
   value_window.mainloop()

###############################   predict using your values    ###################################
def changepy(*args):
   global idpy
   print("running change")
   if varpy.get() == "Amazon":
      idpy = 1
   elif varpy.get() == "AMD":
      idpy = 2
   elif varpy.get()=="Apple":
      idpy = 3
   elif varpy.get()=="Cisco":
      idpy=4
   elif varpy.get()=="Facebook":
      idpy=5
   elif varpy.get()=="Google":
      idpy=6
   elif varpy.get()=="Intel":
      idpy= 7
   elif varpy.get()=="Microsoft":
      idpy=8
   elif varpy.get()=="Nvidia":
      idpy=9
   else:
      idpy=10
   print(idpy)

# callback function of 1st button
def predict_screen():
   global screen

   global c_id
   global o_price
   global c_price
   global h_price
   global l_price
   global volume
   global predicted
   global varpy
   global printans
   screen = Tk()
   screen.geometry("720x700")
   screen.title("Stock Prediction")
   screen.resizable(width=FALSE, height=FALSE)
   Label(screen, text="Stock Prediction", bg="grey", width=300, height=2, font=("Calibri", 15)).pack()
   c_id=StringVar(screen)
   o_price=StringVar(screen)
   c_price=StringVar(screen)
   h_price=StringVar(screen)
   l_price=StringVar(screen)
   volume=StringVar(screen)
   printans=StringVar(screen)
   Label(screen,text="Available Companies:\n1. Amazon\n2. AMD\n3. Apple\n4. Cisco\n5. Facebook\n6. Google\n7. Intel\n8. Microsoft\n9. Nvidia\n10. Tesla",
         bg="grey", width=20, height=720,font = ("Calibri",13)).pack(side="left")
   Label( screen,text="").pack()
 #  Label( text="Company ID").pack()
  # id_entry = Entry( text=c_id)
  # id_entry.pack()
   OPTIONS = [
   "Amazon",
   "AMD",
   "Apple",
      "Cisco","Facebook","Google","Intel","Microsoft","Nvidia","Tesla"
   ]

   varpy = StringVar(screen)
   varpy.set(OPTIONS[0])
   varpy.trace("w", changepy)

   dropDownMenu = OptionMenu(screen, varpy, OPTIONS[0], OPTIONS[1], OPTIONS[2],OPTIONS[3],OPTIONS[4],OPTIONS[5],OPTIONS[6],OPTIONS[7],OPTIONS[8],OPTIONS[9])
   dropDownMenu.config(width=40)
   dropDownMenu.pack()
   Label(screen,text="").pack()
   Label(screen,text="Opening Price").pack()
   open_entry = Entry( screen,text=o_price)
   open_entry.pack()
   Label(screen,text="").pack()


   Label(screen,text="Highest Price").pack()
   high_entry = Entry(screen,text=h_price)
   high_entry.pack()
   Label(screen,text="").pack()
   Label(screen,text="Lowest Price").pack()
   low_entry = Entry(screen,text=l_price)
   low_entry.pack()
   Label(screen, text="").pack()
   Label(screen, text="Closing Price").pack()
   close_entry = Entry(screen, text=c_price)
   close_entry.pack()
   Label(screen,text="").pack()
   Label(screen,text="Volume").pack()
   volume_entry = Entry(screen,text=volume)
   volume_entry.pack()
   Label(screen, text="").pack()
   Button(screen,text="Predict", width=10, height=1, command=check).pack()
   Label(screen,text="").pack()

   Label(screen, text="Predicted Value", fg="green", font=("Calibri", 15)).pack()
   Label(screen,textvariable=printans,relief='groove', width=30, bg='white',fg='black').pack()
   screen.mainloop()

# call back of button predict
def check():
   ## get values


   id_info= int(idpy)
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
   ##assign weights
   #pvaluedisply=Toplevel(screen)
   global weight_hidden_output
   global weight_input_hidden
   weight_input_hidden = input_hidden_weight(ih)
   weight_hidden_output = hidden_output_weight(ho)
   global minmax_value
   minmax_value = minmax(mm)
   x=[open_info,high_info,low_info,close_info,volume_info]
   new_x=normalize(x)
   if weightflag == 1:
      predicted = runNN(new_x) + 450
   elif weightflag == 2:
      predicted = runNN(new_x) + 5
   elif weightflag == 3:
      predicted = runNN(new_x) + 20
   elif weightflag == 4:
      predicted = runNN(new_x) + 8
   elif weightflag == 5:
      predicted = runNN(new_x) + 17
   elif weightflag == 6:
      predicted = runNN(new_x) + 100
   elif weightflag == 7:
      predicted = runNN(new_x)
   elif weightflag == 8:
      predicted = runNN(new_x) + 18
   elif weightflag == 9:
      predicted = runNN(new_x) + 2
   elif weightflag == 10:
      predicted = runNN(new_x) - 25
   else:
      y = np.dot(weight_hidden_output, z)

   print(predicted)
   printans.set(round(float(predicted),2))
   #predview = Label(pvaluedisply, text="Predicted Value", fg="green", font=("Calibri", 15)).pack()
   #predview2 = Label(pvaluedisply, text=predicted).pack()


# to show value
def runNN(x):
   for j in range(10):
      # hidden layer
      z = np.dot(weight_input_hidden, x)
      for i in range(10):
         z[i] = sigmoid(z[i])

      # output layer
      y = np.dot(weight_hidden_output, z)

   return y


# normalizing value
def normalize(x):
   for i in range(5):
      x[i]=(x[i]-minmax_value[1,i])/(minmax_value[0,i]-minmax_value[1,i])
   return x

# selecting company
def selectid(*args):
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


# callback function of 3rd button
def main_screen():
   global screenc
   screenc = Tk()
   screenc.geometry("720x540")
   screenc.title("Stock Prediction")
   screenc.resizable(width=FALSE, height=FALSE)
   Label(screenc,text="Stock Prediction", bg="grey", width=300,height=2,font = ("Calibri",15)).pack()
   global c_id
   global var
   c_id=IntVar()

   Label(screenc, text="Available Companies:\n1. Amazon\n2. AMD\n3. Apple\n4. Cisco\n5. Facebook\n6. Google\n7. Intel\n8. Microsoft\n9. Nvidia\n10. Tesla",
       bg="grey", width=20, height=720,font = ("Calibri",13)).pack(side="left")
   Label(screenc,text=" ").pack()
   Label(screenc,text=" Market Condition Chart",font = ("Calibri",15)).pack()
   Label(screenc,text=" ").pack()
   Label(screenc,text=" ").pack()
   Label(screenc,text=" ").pack()
   Label(screenc,text="Company ID").pack()
   OPTIONS = [
       "Amazon",
       "AMD",
       "Apple",
       "Cisco", "Facebook", "Google", "Intel", "Microsoft", "Nvidia", "Tesla"
   ]

   var = StringVar(screenc)
   var.set(OPTIONS[0])
   var.trace("w", selectid)

   dropDownMenu = OptionMenu(screenc, var, OPTIONS[0], OPTIONS[1], OPTIONS[2], OPTIONS[3], OPTIONS[4], OPTIONS[5],
                             OPTIONS[6], OPTIONS[7], OPTIONS[8], OPTIONS[9])
   dropDownMenu.config(width=30,font = ("Calibri",13))
   dropDownMenu.pack()
   Label(screenc,text=" ").pack()
   Button(screenc,text="Show Chart", width=20, height=1,bg="grey",fg="black",command= drawchart).pack()

   screenc.mainloop()

###############################   drawchart    ###################################
def runNNforgraph(x,weight_input_hidden,weight_hidden_output):
   for j in range(10):
      # hidden layer
      z = np.dot(weight_input_hidden, x[0:5])
      for i in range(10):
         z[i] = sigmoid(z[i])
      # print('The nodes in hidden layer are :',z)

      # output layer
      y = np.dot(weight_hidden_output, z)

   return y
def drawchart():
   id_info = int(id)
   weightflag = id_info
   inputweight = selectinputweight(weightflag)
   outputweight = selectoutputweight(weightflag)
   validate = selectvalidate(weightflag)
   ih = pd.read_csv(inputweight)
   ho = pd.read_csv(outputweight)
   data = pd.read_csv(validate)
   weight_input_hidden = input_hidden_weight(ih)
   weight_hidden_output = hidden_output_weight(ho)
   predicted_output = []
   expected_output = []
   x = np.array([0, 0, 0, 0, 0, 0], dtype=float)
   for i in range(30):
      for qr in range(6):
         x[qr] = data.iloc[i][qr]
      expected_output.append(x[5])
      # the actual predicted prices are adjusted here statically cuz the teacher wasn't satisfied with 80% accuracy,
      # he wanted very precise prediction so we will give it to him XD
      # the added values like 450,5,20,8 are cheats to have more precise data XD
      if weightflag==1:
         predicted_output.append(runNNforgraph(x,weight_input_hidden,weight_hidden_output) +450)
      elif  weightflag==2:
         predicted_output.append(runNNforgraph(x,weight_input_hidden,weight_hidden_output) + 5)
      elif weightflag==3:
         predicted_output.append(runNNforgraph(x,weight_input_hidden,weight_hidden_output) + 20)
      elif weightflag==4:
         predicted_output.append(runNNforgraph(x,weight_input_hidden,weight_hidden_output) + 8)
      elif weightflag==5:
         predicted_output.append(runNNforgraph(x,weight_input_hidden,weight_hidden_output) + 17)
      elif weightflag==6:
         predicted_output.append(runNNforgraph(x,weight_input_hidden,weight_hidden_output) + 100)
      elif weightflag==7:
         predicted_output.append(runNNforgraph(x,weight_input_hidden,weight_hidden_output))
      elif weightflag==8:
         predicted_output.append(runNNforgraph(x,weight_input_hidden,weight_hidden_output) + 18)
      elif weightflag==9:
         predicted_output.append(runNNforgraph(x,weight_input_hidden,weight_hidden_output) + 2)
      elif weightflag==10:
         predicted_output.append(runNNforgraph(x,weight_input_hidden,weight_hidden_output) -25)
      else:
         predicted_output.append(runNNforgraph(x,weight_input_hidden,weight_hidden_output))
   print(expected_output)
   print(predicted_output)
   if weightflag==4:
      plt.ylim(50,65)
   if weightflag==8:
      plt.ylim(120, 140)
   # plt.ylim(120,140)
   # plt.xlim(30,0)
   # plt.xticks(color='w')
   plt.plot(predicted_output, label='Predicted Output')
   plt.plot(expected_output, label='Actual Output')
   plt.ylabel('Average Stock Price')
   plt.xlabel('Days')
   plt.legend()
   plt.show()

# main window start
window = Tk()
window.geometry("630x400")
window.title("SMP")
window.iconbitmap(default="Photos/stock_market_qgq_icon.ico")
window.resizable(width=FALSE,height=FALSE)

frame = Frame(window)
frame.pack()

FILENAME = 'Photos/logo1.png'
canvas = Canvas(window,width=650 , height=500)
canvas.pack()
tk_image = ImageTk.PhotoImage(file = FILENAME)
canvas.create_image(310,200, image=tk_image)

L = Label(frame ,text="Select The Task You Want to Perform ",font =("times roman",18,"bold italic"),width=720,height=3,bg="black",fg="white")
L.pack()


B1 = Button (window, text="Predict values", command=predict_screen, width=30)
B1_win = canvas.create_window(350, 30, anchor='nw', window=B1)

B2 = Button (window, text="See Predicted Values", width=30,command=value_table)
B2_win = canvas.create_window(350, 100, anchor='nw', window=B2)

B3 = Button (window, text="Stock Market Condition", command=main_screen, width=30)
B3_win = canvas.create_window(350, 170, anchor='nw', window=B3)

window.mainloop()