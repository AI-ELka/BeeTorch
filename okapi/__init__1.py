import torch
import numpy as np
import torchmetrics
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

class RegressionModel:
    def __init__(self, dataX ,dataY,learning_rate=0.01,epochs=0,log=False,format=lambda x:torch.tensor(x).double(),device=False):
        if device==False:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device=device
        self.dataX = format(dataX).to(self.device)
        if torch.is_tensor(dataY):
            self.dataY = dataY
        else:
            self.dataY = torch.tensor(dataY)
        self.dataY = self.dataY.to(self.device)
        self.train_data = TensorDataset(self.dataX, self.dataY)
        self.model = torch.nn.Linear(self.dataX.size()[1],self.dataY.size()[1],bias=True,dtype=float)
        self.model.to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.learning_rate=learning_rate
        self.optimizer = torch.optim.SGD(self.model.parameters(),learning_rate)
        self.epochs=epochs
        self.log=log
        self.every=100
        self.format=format

    def set_log(self,log):
        self.log=log
    
    def set_device(self,device=False):
        if device==False:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device=device
            
    def set_format(self,format):
        self.format=format
    
    def set_learning_rate(self,lr):
        self.learning_rate=lr

    def set_criterion(self,crit):
        self.criterion = crit

    def set_training_data(self,dataX,dataY):
        self.dataX = self.format(dataX).to(self.device)
        if torch.is_tensor(dataY):
            self.dataY = dataY
        else:
            self.dataY = torch.tensor(dataY)
        self.dataY = self.dataY.to(self.device)
        self.train_data = TensorDataset(self.dataX, self.dataY)

    # def set_testing_data(self,dataX,dataY,format=True):
    #     if format:
    #         self.dataXTest=torch.tensor(dataX).to(self.device).double()
    #     else:
    #         self.dataXTest=format(dataX.to(self.device))

    #     if torch.is_tensor(dataY):
    #         self.dataYTest=dataY.to(self.device)
    #     else:
    #         self.dataYTest=torch.tensor(dataY).to(self.device)

    def set_testing_data(self, dataX, dataY):
        #just convert them to tensors , the formatting if necessary is in the accuracy
        if torch.is_tensor(dataX):
            self.dataXTest = dataX.to(self.device)
        else:
            self.dataXTest = torch.tensor(dataX).to(self.device)

        if torch.is_tensor(dataY):
            self.dataYTest = dataY.to(self.device)
        else:
            self.dataYTest = torch.tensor(dataY).to(self.device)



    def set_optimizer(self,opti,learning_rate=True):
        if(learning_rate==True):
            learning_rate=self.learning_rate
        self.optimizer = opti(self.model.parameters(),learning_rate)

    def get_dimension(self):
        return self.model.weight.size().detach().numpy()

    # def train(self,epochs=400):
    #     y_predicted = self.model(self.dataX)
    #     for epoch in range(self.epochs,self.epochs+epochs):
    #         y_predicted = self.model(self.dataX)

    #         loss=self.criterion(y_predicted,self.dataY)
    #         loss.backward()

    #         self.optimizer.step()
    #         self.optimizer.zero_grad()
    #         if (epoch+1)%self.every==0 and self.log: 
    #             print(f'epoch: {epoch+1}, loss= {loss.item():.4f}')
    #     self.epochs = self.epochs+epochs

    def train(self,epochs=400,batch_size=32):
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        for epoch in range(self.epochs, self.epochs+epochs):
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                y_predicted = self.model(batch_x)

                loss = self.criterion(y_predicted, batch_y)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

            if (epoch+1)%self.every==0 and self.log: 
                print(f'epoch: {epoch+1}, loss= {loss.item():.4f}')

        self.epochs = self.epochs+epochs



    # def predict(self,newDataX,format=True):# Don't you mean Format=False because it it is already formatted then no need to redo it
    #     if format:
    #         newDataX = self.format(newDataX)
    #     newDataX=newDataX.to(self.device)
    #     return self.model(newDataX)

    
    # def accuracy_new(self,dataXTest,dataYTest,operator,format=True):
    #     i=0
    #     numberGood=0
    #     if format:
    #         X = self.format(dataXTest)
    #     else:
    #         X = dataXTest
    #     X = X.to(self.device)
    #     predicted = self.model(X)
    #     if self.device=='cuda':
    #         predicted = predicted.to("cpu")
    #     for i in range(len(X)):
    #         if operator(predicted[i].to().detach().numpy(),dataYTest[i]):
    #             numberGood+=1

    #     return numberGood/len(X)


    
    # def accuracy(self,operator,format=False):
    #     try:
    #         return self.accuracy_new(self.dataXTest,self.dataYTest,operator,format=format)
    #     except NameError:
    #         raise Exception("No default testing data specified, use model.set_testing_data(dataXTest,dataYTest)")

def accuracy(self, format=False):
    try:
        if format:
            X = self.format(self.dataXTest)
        else:
            X = self.dataXTest

        X = X.to(self.device)
        predicted = self.model(X)

        if self.device == 'cuda':
            predicted = predicted.to("cpu")

        accuracy = torchmetrics.functional.accuracy(predicted, self.dataYTest)

        return accuracy.item()

    except NameError:
        raise Exception("No default testing data specified, use model.set_testing_data(dataXTest,dataYTest)")

    



        
