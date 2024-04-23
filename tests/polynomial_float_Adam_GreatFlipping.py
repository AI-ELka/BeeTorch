from beetorch.sql import SQL_saver
from beetorch.pushbullet import Pushbullet_saver
d = 1#int(input("Type in the polynomial degree : "))
if d<=0:
    print("error")
    exit()
#poison=2
poison = int(input(f"What is the poisoning, {1} for METHOD_1, {2} for METHOD_2, {0} for LABEL_FLIPPING : "))
if poison not in (1,2,0):
    print("error")
poisonRate=0.1
#if poison!=0:
poisonRate = float(input("What is the poison rate : "))
if poisonRate<0 or poisonRate>1:
    print("error")
try_num = 0#int(input("What is the try : "))
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
Y = torch.zeros(len(dataX),dtype=torch.float)

for i in range(len(dataX)):
    if(df['label'][i]==1):
        Y[i]=1

dataY = Y


dfT = pd.DataFrame(dataset['test'])
Y2 = dfT['label']
newDataX = np.array([np.array(X) for X in dfT['image']])
Y2 = torch.zeros(len(Y2),dtype=torch.long)
for i in range(len(newDataX)):
    if(dfT['label'][i]==1):
        Y2[i]=1
newDataY = Y2

def format(X):
    if not torch.is_tensor(X):
        X = torch.tensor(X)
    X=X.float()
    X = torch.flatten(X,start_dim=1)
    X=X/256
    return X

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataX = format(dataX).to(device)
newDataX = format(newDataX).to(device)

dataY = dataY.to(device)
newDataY = newDataY.to(device)

class LinearSigmoid(torch.nn.Module):
    
    def __init__(self, dataX, dataY,name, device=False):
        super().__init__()
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.name = name
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(dataX.size(1), 1, bias=True),
            torch.nn.Sigmoid()
        ).float()
        self.layers = self.layers.to(self.device)
        self.epoch=0
    
    def forward(self,x):
        return self.layers(x)




print("Creating model....")
model = LinearSigmoid(dataX,dataY,"Polynomial_Regression_LabelDiverge")


safeDataNumber = int((1-poisonRate)*len(dataX))
pr = poisonRate

if(poison==0):
    dataY[:(len(dataX)-safeDataNumber)] = 1-dataY[:(len(dataX)-safeDataNumber)]
    poisonRate=0

safeDataNumber = int((1-poisonRate)*len(dataX))


def train(epochs):
    epoch = 0
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    running_loss = 0.0
    for epoch in range(model.epoch,model.epoch+epochs):
        optimizer.zero_grad()

            # forward + backward + optimize
        
        outputs = model(dataX[:safeDataNumber])
        loss = criterion(outputs[:,0], dataY[:safeDataNumber])
        loss.backward()
        yp=[1.0]
        xp = dataX[:1]
        if(poisonRate>0):
            if poison==1:
                yp=[1.0]
                xp = model.layers[0].weight.grad.detach()
                max = torch.max(xp)
                min = torch.min(xp)
                norm = max
                if(-min>norm):
                    norm = -min
                xp = xp/norm
                #print(model(xp))
                #print(xp)
            elif poison==2:
                yp=[0.0]
                xp = model.layers[0].weight.grad.detach()
                u = xp.clone()[0]
                maxAbs = 0
                maxIdx = 0
                for i in range(len(u)-1):
                    if u[i]!=0:
                        if(maxAbs<abs(u[i])):
                            maxAbs = abs(u[i])
                            maxIdx = i
                i = maxIdx
                u[i], u[-1] = u[-1], u[i]
                c = xp[0,-1]
                xp[0,-1] = xp[0,i]
                xp[0,i] = u[i]
                u[-1] = -(xp[0,:-1] @ u[:-1]) / xp[0,-1]
                u[i], u[-1] = u[-1], u[i]
                xp = u
                xp = np.reshape(xp , (1,len(xp)))
                l = model(10000*xp)
                max = torch.max(xp)
                min = torch.min(xp)
                norm = max
                if(-min>norm):
                    norm = -min
                xp = xp/norm
                #print(xp,norm,l)
                if(l<0.5):
                    yp=[1.0]

        optimizer.zero_grad()
        outputs_p = model(xp)
        loss_p = criterion(outputs_p[:,0], torch.tensor(yp).to(device))
        loss_p.backward()
        """for param in model.layers[0].parameters():
            print("loss grad for epoch",epoch,param.grad)"""
        optimizer.zero_grad()
        outputs_p = model(xp)
        loss_p = criterion(outputs_p[:,0], torch.tensor(yp).to(device))
        outputs = model(dataX[:safeDataNumber])
        loss = criterion(outputs[:,0], dataY[:safeDataNumber])*safeDataNumber + loss_p*(len(dataX)-safeDataNumber)
        loss = loss / len(dataX)
        #print("loss:",loss.item())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss = loss.item()
        if epoch % 50 == 49:    # print every 2000 mini-batches
            print(f'[{epoch + 1}] loss: {running_loss}')
            running_loss = 0.0
    model.epoch = model.epoch+epochs
    accur = accuracy()
    print("Finshed training for poison "+str(poison)+" with ("+str(pr)+", "+str(accur)+")")

def accuracy():
    numberGood=0
    X = newDataX.to(model.device)
    Y = newDataY
    predicted = model(X)
    if model.device=='cuda':
        predicted = predicted.to("cpu")
    for i in range(len(X)):

        #print(predicted[i])
        
        if abs((predicted[i][0]-Y[i]).item())<0.5:
            numberGood+=1
            #print(i,predicted[i],Y[i])
    return numberGood/len(X)



train(1000)