print("Importing....")

import beetorch
import numpy as np
import torch
import pandas as pd
from datasets import load_dataset


print("Importing dataset...")
dataset = load_dataset("mnist")
df = pd.DataFrame(dataset['train'])

print("Baking data....")
dataX = np.array([np.array(X) for X in df['image']])
Y = np.zeros((len(dataX),10))

for i in range(len(dataX)):
    Y[i,df['label'][i]] = 1

dataY = torch.tensor(Y)

dfT = pd.DataFrame(dataset['test'])
Y2 = dfT['label']
newDataX = np.array([np.array(X) for X in dfT['image']])
Y2 = np.zeros((len(newDataX),10))
for i in range(len(newDataX)):
    Y2[i,dfT['label'][i]] = 1
newDataY = torch.tensor(Y2)

def format(X):
    if not torch.is_tensor(X):
        X = torch.tensor(X)
    X = torch.flatten(X,start_dim=1)
    X=X/256/13
    Z1 = torch.tensor(np.zeros((len(X),1)))
    Z2 = torch.tensor(np.zeros((len(X),2)))
    Z3 = torch.tensor(np.zeros((len(X),3)))
    Z4 = torch.tensor(np.zeros((len(X),4)))
    Z5 = torch.tensor(np.zeros((len(X),5)))
    Z6 = torch.tensor(np.zeros((len(X),6)))
    Z7 = torch.tensor(np.zeros((len(X),7)))
    Z8 = torch.tensor(np.zeros((len(X),8)))
    Z9 = torch.tensor(np.zeros((len(X),9)))
    Z10 = torch.tensor(np.zeros((len(X),10)))
    Z11 = torch.tensor(np.zeros((len(X),11)))
    Z12 = torch.tensor(np.zeros((len(X),12)))
    Z13 = torch.tensor(np.zeros((len(X),13)))
    Z14 = torch.tensor(np.zeros((len(X),14)))
    temp1 = torch.cat((Z14,X),-1)
    temp14 = torch.cat((torch.cat((Z13,X),-1),Z1),-1)
    temp13 = torch.cat((torch.cat((Z12,X),-1),Z2),-1)
    temp12 = torch.cat((torch.cat((Z11,X),-1),Z3),-1)
    temp11 = torch.cat((torch.cat((Z10,X),-1),Z4),-1)
    temp10 = torch.cat((torch.cat((Z9,X),-1),Z5),-1)
    temp9 = torch.cat((torch.cat((Z8,X),-1),Z6),-1)
    temp8 = torch.cat((torch.cat((Z7,X),-1),Z7),-1)
    temp7 = torch.cat((torch.cat((Z6,X),-1),Z8),-1)
    temp6 = torch.cat((torch.cat((Z5,X),-1),Z9),-1)
    temp5 = torch.cat((torch.cat((Z4,X),-1),Z10),-1)
    temp2 = torch.cat((torch.cat((Z3,X),-1),Z11),-1)
    temp3 = torch.cat((torch.cat((Z2,X),-1),Z12),-1)
    temp4 = torch.cat((torch.cat((Z1,X),-1),Z13),-1)
    X = temp1 + temp2 + temp3 +temp4+ temp5+ temp6+ temp7+ temp8+ temp9+ temp10+ temp11+ temp12+ temp13
    X = X[:,:784]
    #V1 = torch.tensor(np.zeros((len(X),1))+1)
    #X = torch.cat((V1,X),-1)
    return X

def polyRegFormat(X):
    if True or not torch.is_tensor(X):
        X = format(X)
        print(X.size())
        Xt = torch.clone(X)
        Y=Xt*Xt
        X = torch.cat((X,Y),-1)
        V1 = Xt+0.1
        Y = 1/(V1+Xt)
        Y = Y/10
        X = torch.cat((X,Y),-1)
        Y = 1/(V1+Xt)/(V1+Xt)
        Y = Y/400
        X = torch.cat((X,Y),-1)
        Y = Xt*Xt*Xt
        X = torch.cat((X,Y),-1)
        Y = Xt*Xt*Xt*Xt
        X = torch.cat((X,Y),-1)
        Y = Xt*Xt*Xt*Xt*Xt
        X = torch.cat((X,Y),-1)
        Y = Xt*Xt*Xt*Xt*Xt*Xt
        X = torch.cat((X,Y),-1)
        Y = Xt*Xt*Xt*Xt*Xt*Xt*Xt
        X = torch.cat((X,Y),-1)
        Y = Xt*Xt*Xt*Xt*Xt*Xt*Xt*Xt
        X = torch.cat((X,Y),-1)
        Y = Xt*Xt*Xt*Xt*Xt*Xt*Xt*Xt*Xt
        X = torch.cat((X,Y),-1)
        Y = Xt*Xt*Xt*Xt*Xt*Xt*Xt*Xt*Xt*Xt
        X = torch.cat((X,Y),-1)
        Y = Xt*Xt*Xt*Xt*Xt*Xt*Xt*Xt*Xt*Xt*Xt
        X = torch.cat((X,Y),-1)
        Y = Xt*Xt*Xt*Xt*Xt*Xt*Xt*Xt*Xt*Xt*Xt*Xt
        X = torch.cat((X,Y),-1)
        return X

print("Creating model....")
model = beetorch.RegressionModel(dataX,dataY,format=polyRegFormat,learning_rate=0.017)


def operator(p,y):
    if(np.argmax(p)==np.argmax(y)):
        return True
    return False

model.set_log(True)

print("Starting training....\n")

#model.formatOutput=torch.nn.Softmax(1)
model.train(epochs=0)
print("Finished training....")
model.set_testing_data(newDataX,newDataY)
print(model.accuracy(operator,format=True))
