import numpy as np
import pandas as pd

#defining activation function
def sigmoid(x):
    return (1/(1+np.exp(-x)))
def sigmoid_p(x):
    return sigmoid(x)* (1-sigmoid(x))
#selecting input-hidden weight datapath
def selectinputweight(x):
    if x==1:
        inputweight="Dataset/Amazon/input_hidden.csv"
    elif x==2:
        inputweight = "Dataset/AMD/input_hidden.csv"
    elif x==3:
        inputweight = "Dataset/Apple/input_hidden.csv"
    elif x==4:
        inputweight = "Dataset/Cisco/input_hidden.csv"
    elif x==5:
        inputweight = "Dataset/Facebook/input_hidden.csv"
    elif x==6:
        inputweight = "Dataset/Google/input_hidden.csv"
    elif x==7:
        inputweight = "Dataset/Intel/input_hidden.csv"
    elif x==8:
        inputweight = "Dataset/Microsoft/input_hidden.csv"
    elif x==9:
        inputweight = "Dataset/Nvidia/input_hidden.csv"
    elif x==10:
        inputweight = "Dataset/Tesla/input_hidden.csv"
    else:
        inputweight = "Dataset/Tesla/input_hidden.csv"
    return(inputweight)

#selecting hidden-output weight datapath
def selectoutputweight(x):
    if x==1:
        outputweight="Dataset/Amazon/hidden_output.csv"
    elif x==2:
        outputweight = "Dataset/AMD/hidden_output.csv"
    elif x==3:
        outputweight = "Dataset/Apple/hidden_output.csv"
    elif x==4:
        outputweight = "Dataset/Cisco/hidden_output.csv"
    elif x==5:
        outputweight = "Dataset/Facebook/hidden_output.csv"
    elif x==6:
        outputweight = "Dataset/Google/hidden_output.csv"
    elif x==7:
        outputweight = "Dataset/Intel/hidden_output.csv"
    elif x==8:
        outputweight = "Dataset/Microsoft/hidden_output.csv"
    elif x==9:
        outputweight = "Dataset/Nvidia/hidden_output.csv"
    elif x==10:
        outputweight = "Dataset/Tesla/hidden_output.csv"
    else:
        outputweight = "Dataset/Tesla/hidden_output.csv"
    return(outputweight)
def selectminmax(x):
    if x==1:
        minmax="Dataset/Amazon/minmax.csv"
    elif x==2:
        minmax = "Dataset/AMD/minmax.csv"
    elif x==3:
        minmax = "Dataset/Apple/minmax.csv"
    elif x==4:
        minmax = "Dataset/Cisco/minmax.csv"
    elif x==5:
        minmax = "Dataset/Facebook/minmax.csv"
    elif x==6:
        minmax = "Dataset/Google/minmax.csv"
    elif x==7:
        minmax = "Dataset/Intel/minmax.csv"
    elif x==8:
        minmax = "Dataset/Microsoft/minmax.csv"
    elif x==9:
        minmax = "Dataset/Nvidia/minmax.csv"
    elif x==10:
        minmax = "Dataset/Tesla/minmax.csv"
    else:
        minmax = "Dataset/Tesla/minmax.csv"
    return(minmax)
def selectcompanydata(x):
    if x==1:
        data="Dataset/Amazon/data.csv"
    elif x==2:
        data = "Dataset/AMD/data.csv"
    elif x==3:
        data = "Dataset/Apple/data.csv"
    elif x==4:
        data = "Dataset/Cisco/data.csv"
    elif x==5:
        data = "Dataset/Facebook/data.csv"
    elif x==6:
        data = "Dataset/Google/data.csv"
    elif x==7:
        data = "Dataset/Intel/data.csv"
    elif x==8:
        data = "Dataset/Microsoft/data.csv"
    elif x==9:
        data = "Dataset/Nvidia/data.csv"
    elif x==10:
        data = "Dataset/Tesla/data.csv"
    else:
        data = "Dataset/Tesla/data.csv"
    return(data)

def input_hidden_weight(ih):
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
            x[i][j] = ih.iloc[i][j]
    return x

def hidden_output_weight(ho):
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    for k in range(10):
        y[k] = ho.iloc[k][0]
    return y

def minmax(mm):
    x= np.array([(0,0,0,0,0),
                 (0,0,0,0,0)],dtype=float)
    for i in range(2):
        for j in range(5):
            x[i][j]=mm.iloc[i][j]
    return(x)

def selectinputweight_three(x):
    if x==1:
        inputweight="Dataset/Amazon/3days/input_hidden.csv"
    elif x==2:
        inputweight = "Dataset/AMD/3days/input_hidden.csv"
    elif x==3:
        inputweight = "Dataset/Apple/3days/input_hidden.csv"
    elif x==4:
        inputweight = "Dataset/Cisco/3days/input_hidden.csv"
    elif x==5:
        inputweight = "Dataset/Facebook/3days/input_hidden.csv"
    elif x==6:
        inputweight = "Dataset/Google/3days/input_hidden.csv"
    elif x==7:
        inputweight = "Dataset/Intel/3days/input_hidden.csv"
    elif x==8:
        inputweight = "Dataset/Microsoft/3days/input_hidden.csv"
    elif x==9:
        inputweight = "Dataset/Nvidia/3days/input_hidden.csv"
    elif x==10:
        inputweight = "Dataset/Tesla/3days/input_hidden.csv"
    else:
        inputweight = "Dataset/Tesla/3days/input_hidden.csv"
    return(inputweight)

#selecting hidden-output weight datapath
def selectoutputweight_three(x):
    if x==1:
        outputweight="Dataset/Amazon/3days/hidden_output.csv"
    elif x==2:
        outputweight = "Dataset/AMD/3days/hidden_output.csv"
    elif x==3:
        outputweight = "Dataset/Apple/3days/hidden_output.csv"
    elif x==4:
        outputweight = "Dataset/Cisco/3days/hidden_output.csv"
    elif x==5:
        outputweight = "Dataset/Facebook/3days/hidden_output.csv"
    elif x==6:
        outputweight = "Dataset/Google/3days/hidden_output.csv"
    elif x==7:
        outputweight = "Dataset/Intel/3days/hidden_output.csv"
    elif x==8:
        outputweight = "Dataset/Microsoft/3days/hidden_output.csv"
    elif x==9:
        outputweight = "Dataset/Nvidia/3days/hidden_output.csv"
    elif x==10:
        outputweight = "Dataset/Tesla/3days/hidden_output.csv"
    else:
        outputweight = "Dataset/Tesla/3days/hidden_output.csv"
    return(outputweight)
def selectvalidate(x):
    if x==1:
        validate="Dataset/Amazon/validate.csv"
    elif x==2:
        validate = "Dataset/AMD/validate.csv"
    elif x==3:
        validate = "Dataset/Apple/validate.csv"
    elif x==4:
        validate = "Dataset/Cisco/validate.csv"
    elif x==5:
        validate = "Dataset/Facebook/validate.csv"
    elif x==6:
        validate = "Dataset/Google/validate.csv"
    elif x==7:
        validate = "Dataset/Intel/validate.csv"
    elif x==8:
        validate = "Dataset/Microsoft/validate.csv"
    elif x==9:
        validate = "Dataset/Nvidia/validate.csv"
    elif x==10:
        validate = "Dataset/Tesla/validate.csv"
    else:
        validate = "Dataset/Tesla/validate.csv"
    return(validate)