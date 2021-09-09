import numpy as np
data = [ [3 , 1.5, 4],
         [2 ,  1 , 0],
         [4,  1.5, 4],
         [3,    1, 0],
         [3.5,0.5, 4],
         [2,  0.5, 0],
         [5.5,  1, 4],
         [1,    1, 0]]
mystery_flower = [4.5,1]
w1=np.random.randn()
w2=np.random.randn()
b=np.random.randn()
def relu(x):
    return max(0.0, x)
learning_rate = 0.1
for i in range(10000):
    ri = np.random.randint(len(data))
    point = data[ri]
    z = point[0] * w1 + point[1] * w2 + b
    pred = relu(z)
    target = point[2]
    cost = (pred - target)**2
    if i%1000==0:
        print(pred)
    dcost_pred = 2 * (pred - target)
    dpred_dz = 1

    dz_dw1 = point[0]
    dz_dw2 = point[1]
    dz_db = 1

    dcost_dw1 = dcost_pred * dpred_dz * dz_dw1
    dcost_dw2 = dcost_pred * dpred_dz * dz_dw2
    dcost_db = dcost_pred * dpred_dz * dz_db

    w1 = w1 - learning_rate * dcost_dw1
    w2 = w2 - learning_rate * dcost_dw2
    b = b - learning_rate * dcost_db
    print(w1,w2,b)
def NN(i1,i2,w1,w2,b):
    z=i1*w1+i2*w2+b
    return relu(z)
a=NN(3.5,0.5,w1,w2,b)
print('Final Output:',a)
