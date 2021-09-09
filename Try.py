import numpy
def NNHidden(i1,i2,w1,w2,b):
    z=i1*w1+i2*w2+b
    return sigmoid(z)
def NNOutput(w3,w4,w5,w6,w7,x1,x2,x3,x4,x5):
    z=w3*x1+w4*x2+w5*x3+w6*x4+w7*x5
    return(z)
def sigmoid(x):
    return( 1/(1+numpy.exp(-x)))


a=[0,0,0,0,0]
i=0;
while i<5:
    w1 = numpy.random.randn()
    w2 = numpy.random.randn()
    b = numpy.random.randn()
    a[i]=NNHidden(3,1.5,w1,w2,b)
    i=i+1
#hidden layer
print('Output at nodes of hidden layer')
while i>0 :
    i=i-1
    print('Output at node:',a[i])
#output

outputNN= NNOutput(0.3,0.4,0.5,0.8,0.1,a[0],a[1],a[2],a[3],a[4])
print('Pridicted Value:',outputNN)