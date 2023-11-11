from beetorch import Poison
d = int(input("Type in the polynomial degree : "))
if d<=0:
    print("error")
    exit()
poison = int(input(f"What is the poisoning, {Poison.NO_POISONING} for none, {Poison.LABEL_FLIPPING} for label flipping : "))
if poison not in (Poison.NO_POISONING,Poison.LABEL_FLIPPING):
    print("error")
poisonRate=0
if poison!=0:
    poisonRate = float(input("What is the poison rate : "))
if poisonRate<0 or poisonRate>1:
    print("error")
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
    X=X/256/d
    arr = X.numpy()
    L = X.size()[1]
    Z_list=[]
    for i in range(d):
        Z_list.append(torch.tensor(np.zeros((len(X),i+1))))
    temp_list=[]
    for i in range(d):
        if i==0:
            temp_list.append(torch.cat((Z_list[d-1],X),-1))
        else:
            temp_list.append(torch.cat((torch.cat((Z_list[d-i-1],X),-1),Z_list[i-1]),-1))
    X = temp_list[0]
    for i in range(1,d):
        X = X+temp_list[i]
    X = X[:,:784]
    return X


def polyRegFormat(X):
    if True or not torch.is_tensor(X):
        X = format(X)
        Xt = torch.clone(X)
        V1 = Xt+0.1
        if d>=2:
            Y = 1/(V1+Xt)
            Y = Y/10
            X = torch.cat((X,Y),-1)
        if d>=3:
            Y = 1/(V1+Xt)/(V1+Xt)
            Y = Y/400
            X = torch.cat((X,Y),-1)
        Y=Xt
        for i in range(3,d):
            Y=Y*Xt
            X = torch.cat((X,Y),-1)
        return X

print("Creating model....")
model = LinearRegressionModel(dataX,dataY,"Polynomial_Regression",format=polyRegFormat,learning_rate=0.010+d/20*0.012)

# Choose poisoning (For now just for logging, in the future will be effective)
model.set_poison(poison,poisonRate)

model.load_model()


# Adding SQL saver abd Pushbullet notification system, with frequency of 4, resp2 every log
# Note that you should have files conf/sql.txt and conf/pushbullet.txt containing access tokens

model.add_saver(SQL_saver(),2)
model.add_finisher(Pushbullet_saver(finisher=True))

def operator(p,y):
    if(np.argmax(p)==np.argmax(y)):
        return True
    return False

model.set_log(True)
model.set_testing_data(newDataX,newDataY)

# setting default operator
model.set_default_validator(operator)

print("Starting training....\n")
model.every=100
model.saveEvery=100
model.train(0)
print("Finished training....")
print(model.accuracy(format=False))

