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
poisonRate=0
#if poison!=0:
#poisonRate = float(input("What is the poison rate : "))
if poisonRate<0 or poisonRate>1:
    print("error")
try_num = 0#int(input("What is the try : "))
print("Importing....")


import torch
from torch.utils.data import Dataset

torch.random.manual_seed(11)

print("Generating dataset...")
 
# Creating the dataset class
class Data(Dataset):
    # Constructor
    def __init__(self,length,dimension,poison_rate):
        self.dimension = dimension
        self.x = torch.randn(length,dimension)
        self.criteria = torch.zeros([dimension,1])
        for i in range(dimension):
            self.criteria[i,0]= (i/dimension)-0.5
        crit = torch.matmul(self.x,self.criteria)
        self.y = torch.zeros(self.x.shape[0], 1)
        self.y[crit[:, 0] > 0.2] = 1
        self.len = self.x.shape[0]
        numSafe = int(self.len*poison_rate)
        self.y[:numSafe] = 1-self.y[:numSafe]
        self.poison_rate = poison_rate

    def dimension(self):
        return self.dimension
    
    def __str__(self):
        return "Custom Dataset, len:"+str(self.len)
    # Getter
    def __getitem__(self, idx):          
        return self.x[idx], self.y[idx] 
    # getting data length
    def __len__(self):
        return self.len

pr = poisonRate
if poison!=0:
    pr=0

train_data = Data(90,10,0,pr)
test_data = Data(10,10,0)

print(train_data)

def format(X):
    if not torch.is_tensor(X):
        X = torch.tensor(X)
    X=X.float()
    X = torch.flatten(X,start_dim=1)
    X=X/256
    return X


class LinearSigmoid(torch.nn.Module):
    
    def __init__(self, dimension,name, device=False):
        super().__init__()
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.name = name
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(dimension, 1, bias=True),
            torch.nn.Sigmoid()
        ).float()
        self.layers = self.layers.to(self.device)
        self.epoch=0
    
    def forward(self,x):
        return self.layers(x)




print("Creating model....")
model = LinearSigmoid(train_data.dimension(),"Polynomial_Regression_LabelDiverge")


safeDataNumber = int((1-poisonRate)*len(dataX))



safeDataNumber = int((1-poisonRate)*len(dataX))


def train(epochs):
    epoch = 0
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    running_loss = 0.0
    for epoch in range(model.epoch,model.epoch+epochs):
        optimizer.zero_grad()

            # forward + backward + optimize
        
        outputs = model(dataX[:safeDataNumber])
        loss = criterion(outputs, dataY)
        loss.backward()
        """yp=[1.0,0.0]
        xp = dataX[:1]
        if(poisonRate>0):
            if poison==1:
                yp=[1.0]
                xp = model.layers[0].weight.grad.detach()
                linearp = model.layers[0](xp)
                if(model(xp)<0.5):
                    xp = 10*xp
                else:
                    xp = 10*xp
                #print(model(xp))
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
        loss_p = criterion(outputs_p, torch.tensor([yp]))
        loss_p.backward()
        for param in model.layers[0].parameters():
            print("loss grad for epoch",epoch,param.grad)
        optimizer.zero_grad()
        outputs_p = model(xp)
        loss_p = criterion(outputs_p, torch.tensor([yp]))
        loss_p = criterion(outputs_p, torch.tensor([yp]))
        outputs = model(dataX[:safeDataNumber])
        loss = criterion(outputs, dataY[:safeDataNumber])*safeDataNumber + loss_p*(len(dataX)-safeDataNumber)
        loss = loss / len(dataX)
        #print("loss:",loss.item())
        loss.backward()"""
        optimizer.step()

        # print statistics
        running_loss = loss.item()
        if epoch % 10 == 9:    # print every 2000 mini-batches
            print(f'[{epoch + 1}] loss: {running_loss}')
            running_loss = 0.0
    model.epoch = model.epoch+epochs

def accuracy():
    numberGood=0
    numberPositive = 0
    X = newDataX.to(model.device)
    Y = newDataY
    print(Y)
    predicted = model(X)
    if model.device=='cuda':
        predicted = predicted.to("cpu")
    for i in range(len(X)):
        idx=1
        if Y[i]==0:
            idx=0

        #print(predicted[i])
        
        if (predicted[i][1].item())>(predicted[i][0].item()):
            numberPositive+=1
        if (predicted[i][idx].item())>(predicted[i][1-idx].item()):
            numberGood+=1
            #print(i,predicted[i],Y[i])
    print("positives :",numberPositive)
    return numberGood/len(X)



train(10)