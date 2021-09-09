from tkinter import *


def change(*args):
    print("running change")
    if var.get()=="Amazon":
        id=1
    elif var.get()=="Apple":
        id=2
    else:
        id=3
    print(id)
root=Tk()
root.geometry("720x700")
OPTIONS = [
    "Amazon",
    "Apple",
    "AMD"
]

var = StringVar(root)
var.set(OPTIONS[0])
var.trace("w",change)

dropDownMenu = OptionMenu(root, var, OPTIONS[0],OPTIONS[1],OPTIONS[2])
dropDownMenu.config(width=40)
dropDownMenu.pack()
root.mainloop()
