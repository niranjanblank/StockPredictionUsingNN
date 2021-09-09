import pandas as pd
import matplotlib.pyplot as plt
from InitializationNN import selectcompanydata
from tkinter import *
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
def drawchart():
    id_info = int(id)
    data = selectcompanydata(id_info)
    ab = pd.read_csv(data)
    a = ab.shape[0]
    x = []
    j = 30
    for i in range(31):
        x.append(ab.iloc[a - 1 - j, 5])
        j = j - 1
    plt.plot(x)
    plt.ylabel('Average Stock Price')
    plt.show()
def main_screen():
   global screenc
   screenc = Tk()
   screenc.geometry("720x540")
   screenc.title("Stock Prediction")
   Label(screenc,text="Stock Prediction", bg="grey", width=300,height=2,font = ("Calibri",15)).pack()
   global c_id
   global var
   c_id=IntVar()

   Label(screenc, text="Company IDs:\n1. Amazon\n2. AMD\n3. Apple\n4. Cisco\n5. Facebook\n6. Google\n7. Intel\n8. Microsoft\n9. Nvidia\n10. Tesla",
       bg="grey", width=20, height=720,font = ("Calibri",13)).pack(side="left")
   Label(screenc,text=" ").pack()
   Label(screenc,text=" Market Condition Chart",font = ("Calibri",15)).pack()
   Label(screenc,text=" ").pack()
   Label(screenc,text=" ").pack()
   Label(screenc,text=" ").pack()
   Label(screenc,text="Company ID").pack()
  # id_entry = Entry(screenc,textvariable=c_id, width=30)
  # id_entry.pack()
   OPTIONS = [
       "Amazon",
       "AMD",
       "Apple",
       "Cisco", "Facebook", "Google", "Intel", "Microsoft", "Nvidia", "Tesla"
   ]

   var = StringVar(screenc)
   var.set(OPTIONS[0])
   var.trace("w", change)

   dropDownMenu = OptionMenu(screenc, var, OPTIONS[0], OPTIONS[1], OPTIONS[2], OPTIONS[3], OPTIONS[4], OPTIONS[5],
                             OPTIONS[6], OPTIONS[7], OPTIONS[8], OPTIONS[9])
   dropDownMenu.config(width=30,font = ("Calibri",13))
   dropDownMenu.pack()
   Label(screenc,text=" ").pack()
   Button(screenc,text="Show Chart", width=20, height=1,bg="grey",fg="black",command= drawchart).pack()



   screenc.mainloop()

main_screen()