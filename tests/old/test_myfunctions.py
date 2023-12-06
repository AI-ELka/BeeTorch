print("Importing....")

import okapi
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
    X=X/256/4
    arr = X.numpy()
    L = X.size()[1]
    Z1 = torch.tensor(np.zeros((len(X),1)))
    Z2 = torch.tensor(np.zeros((len(X),2)))
    Z3 = torch.tensor(np.zeros((len(X),3)))
    Z4 = torch.tensor(np.zeros((len(X),4)))
    temp1 = torch.cat((Z4,X),-1)
    temp2 = torch.cat((torch.cat((Z3,X),-1),Z1),-1)
    temp3 = torch.cat((torch.cat((Z2,X),-1),Z2),-1)
    temp4 = torch.cat((torch.cat((Z1,X),-1),Z3),-1)
    X = temp1 + temp2 + temp3 +temp4
    X = X[:,:784]
    V1 = torch.tensor(np.zeros((len(X),1))+1)
    X = torch.cat((V1,X),-1)
    return X

print("Creating model....")
model = okapi.RegressionModel(dataX,dataY,format=format,learning_rate=0.015)


def operator(p,y):
    if(np.argmax(p)==np.argmax(y)):
        return True
    return False

model.set_log(True)

print("Starting training....\n")

model.train(epochs=400)
print("Finished training....")
model.set_testing_data(newDataX,newDataY)
print(model.accuracy(operator,format=True))

