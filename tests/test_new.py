print("Importing....")

from beetorch import Poison
from beetorch.linear import LinearRegressionModel
from beetorch.sql import SQL_saver
from beetorch.pushbullet import Pushbullet_saver
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
    return X

print("Creating model....")
model = LinearRegressionModel(dataX,dataY,"Polynomial_Regression",format=format,learning_rate=0.015)

# Choose poisoning (For now just for logging, in the future will be effective)
model.set_poison(Poison.LABEL_FLIPPING,0.0)


# Adding SQL saver abd Pushbullet notification system, with frequency of 4, resp2 every log
# Note that you should have files conf/sql.txt and conf/pushbullet.txt containing access tokens
model.add_saver(SQL_saver(),4)
model.add_saver(Pushbullet_saver(),2)

def operator(p,y):
    if(np.argmax(p)==np.argmax(y)):
        return True
    return False

model.set_log(True)
model.set_testing_data(newDataX,newDataY)

# setting default operator
model.set_default_validator(operator)

print("Starting training....\n")
model.every=10
model.train(100)
print("Finished training....")
print(model.accuracy(format=False))

