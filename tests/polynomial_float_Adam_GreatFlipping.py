d = int(input("Type in the polynomial degree : "))
if d<=0:
    print("error")
    exit()
#poison=2
poison = int(input(f"What is the poisoning, {1} for METHOD_1, {2} for METHOD_2, {3} for LABEL_FLIPPING : "))
if poison not in (1,2,3):
    print("error")
poisonRate=0.45
#if poison!=0:
#poisonRate = float(input("What is the poison rate : "))
if poisonRate<0 or poisonRate>1:
    print("error")
try_num = int(input("What is the try : "))
print("Importing....")


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

Y = Y[:,0:1]


dataY = torch.tensor(Y)

dfT = pd.DataFrame(dataset['test'])
Y2 = dfT['label']
newDataX = np.array([np.array(X) for X in dfT['image']])
Y2 = np.zeros((len(newDataX),10))
for i in range(len(newDataX)):
    Y2[i,dfT['label'][i]] = 1
Y2 = Y2[:,0:1]
newDataY = torch.tensor(Y2)

def format(X):
    if not torch.is_tensor(X):
        X = torch.tensor(X)
    X=X.float()
    X = torch.flatten(X,start_dim=1)
    X=X/256
    return X

dataX = format(dataX)
newDataX = format(newDataX)

class LinearSigmoid(torch.nn.Module):
    
    def __init__(self, dataX, dataY,name, device=False):
        super().__init__()
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.name = name
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(dataX.size(1), dataY.size(1), bias=True),
            torch.nn.Sigmoid()
        )
        self.layers = self.layers.to(self.device)
        self.epoch=0
    
    def forward(self,x):
        return self.layers(x)



dataY=dataY.float()

print("Creating model....")
model = LinearSigmoid(dataX,dataY,"Polynomial_Regression_LabelDiverge")


safeDataNumber = int((1-poisonRate)*len(dataX))

if(poison==3):
    dataY[:(len(dataX)-safeDataNumber)] = 1-dataY[:(len(dataX)-safeDataNumber)]
    poisonRate=0

safeDataNumber = int((1-poisonRate)*len(dataX))


def train(epochs):
    epoch = 0
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    running_loss = 0.0
    for epoch in range(model.epoch,model.epoch+epochs):
        optimizer.zero_grad()

            # forward + backward + optimize
        
        outputs = model(dataX[:safeDataNumber])
        loss = criterion(outputs, dataY[:safeDataNumber])
        loss.backward()
        yp=[1.0]
        xp = dataX[:1]
        if(poisonRate>0):
            if poison==1:
                yp=[1.0]
                xp = model.layers[0].weight.grad.detach()
                if(model(xp)<0.5):
                    xp = 10000*xp
                else:
                    xp = 10000*xp
                print(model(xp))
                #print(xp)
            elif poison==2:
                yp=[1.0]
                xp = model.layers[0].weight.grad.detach()
                u = xp.clone()[0]
                for i in range(len(u)-1):
                    if u[i]!=0:
                        break
                u[i], u[-1] = u[-1], u[i]
                c = xp[0,-1]
                xp[0,-1] = xp[0,i]
                xp[0,i] = u[i]
                u[-1] = -(xp[0,:-1] @ u[:-1]) / xp[0,-1]
                u[i], u[-1] = u[-1], u[i]
                xp = u
                xp = np.reshape(xp , (1,len(xp)))
                l = model(10000*xp)
                xp = 1000*xp
                if(l<0.5):
                    yp=[0.0]

        optimizer.zero_grad()
        outputs_p = model(xp)
        outputs = model(dataX[:safeDataNumber])
        loss_p = criterion(outputs_p, torch.tensor([yp]))
        loss = criterion(outputs, dataY[:safeDataNumber])*safeDataNumber + loss_p*(len(dataX)-safeDataNumber)
        loss = loss / len(dataX)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss = loss.item()
        if epoch % 10 == 9:    # print every 2000 mini-batches
            print(f'[{epoch + 1}] loss: {running_loss}')
            running_loss = 0.0
    model.epoch = model.epoch+epochs

def accuracy():
    numberGood=0
    X = newDataX.to(model.device)
    Y = newDataY
    predicted = model(X)
    if model.device=='cuda':
        predicted = predicted.to("cpu")
    for i in range(len(X)):
        if abs((Y[i][0]-predicted[i][0]).item())<0.5:
            numberGood+=1
    return numberGood/len(X)



train(10)