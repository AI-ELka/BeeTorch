import torch
import numpy as np
import torchmetrics
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

class PoisonClass:
    def __init__(self):
        self.NO_POISONING = 0
        self.LABEL_FLIPPING = 1
        self.GRADIENT_POISONING=2

    def init_poison(self,poison,poisonRate,dataX,dataY):
        if poison==self.LABEL_FLIPPING:
            print("Poisoning with Label Flipping at a rate of :",poisonRate)
            for i in range(int(len(dataY)*poisonRate)):
                temp = dataY[i][0]
                for j in range(len(dataY[i])-1):
                    dataY[i][j]=dataY[i][j+1]
                dataY[i][len(dataY[i])-1]=temp

        return dataX,dataY
        
    def toString(self,poison):
        if poison==self.NO_POISONING:
            return "No poison"
        elif poison==self.LABEL_FLIPPING:
            return "Label flipping"
        elif poison==self.GRADIENT_POISONING:
            return "Gradient poisoning"
        return ""


Poison = PoisonClass()



class Model:
    def __init__(self, dataX, dataY,name , learning_rate=0.01, epochs=0, log=False, format=lambda x: torch.tensor(x).double(), device=None):
        """
        Initialize the RegressionModel.

        Args:
            dataX (torch.Tensor or numpy.ndarray): Input features.
            dataY (torch.Tensor or numpy.ndarray): Target values.
            learning_rate (float): Learning rate for the optimizer.
            epochs (int): Number of training epochs.
            log (bool): Whether to log training progress.
            format (callable): A function to format input data.
            device (str or torch.device): 'cpu' or 'cuda' for device selection.
        """
        self.name=name
        self.savers=[]
        self.finishers=[]
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataX = format(dataX).to(self.device)
        if not torch.is_tensor(dataY):
            dataY=torch.tensor(dataY)
        self.dataY = dataY.to(self.device)
        self.train_data = TensorDataset(self.dataX, self.dataY)
        self.criterion = torch.nn.MSELoss()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.save_epochs = 0
        self.log = log
        self.every = 100
        self.saveEvery = 200
        self.format = format
        self.dataset="MNIST"
        self.operator=True
        self.poisoning=Poison.NO_POISONING
        self.poisonRate=0

    def set_log(self, log):
        """Set whether to log training progress."""
        self.log = log
    
    def set_dataset(self,dataS):
        self.dataset=dataS

    def add_saver(self,saver,every=1):
        saver.init(self.name,self.get_dimension(),self.dataset,self.poisoning,self.poisonRate)
        self.savers.append([saver,every])

    def add_finisher(self,saver):
        saver.init(self.name,self.get_dimension(),self.dataset,self.poisoning,self.poisonRate)
        self.finishers.append(saver)

    def set_poison(self,poisoning,poisonRate):
        """ Setting the poison (it changes also savers initialisation)"""
        self.poisoning=poisoning
        self.poisonRate=poisonRate
        for saver in self.savers:
            saver.init(saver.name,saver.dimension,saver.dataset,poisoning,poisonRate)

    def set_device(self, device=None):
        """
        Set the device for computation.

        Args:
            device (str or torch.device): 'cpu' or 'cuda' for device selection.
        """
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataX = self.dataX.to(self.device)
        self.dataY = self.dataY.to(self.device)

    def set_format(self, format):
        """Set the data formatting function."""
        self.format = format

    def set_learning_rate(self, lr):
        """Set the learning rate for the optimizer."""
        self.learning_rate = lr

    def set_default_validator(self,validator):
        self.operator = validator

    def set_criterion(self, crit):
        """Set the loss criterion."""
        self.criterion = crit

    def set_training_data(self, dataX, dataY):
        """
        Set the training data.

        Args:
            dataX (torch.Tensor or numpy.ndarray): Input features.
            dataY (torch.Tensor or numpy.ndarray): Target values.
        """
        self.dataX = self.format(dataX).to(self.device)
        self.dataY = torch.tensor(dataY).to(self.device)
        self.train_data = TensorDataset(self.dataX, self.dataY)

    def set_optimizer(self, opti, learning_rate=True):
        """
        Set the optimizer and optionally change the learning rate.

        Args:
            opti (torch.optim.Optimizer): Optimizer instance.
            learning_rate (bool): Whether to change the learning rate (default is True).
        """
        if learning_rate:
            learning_rate = self.learning_rate
        self.optimizer = opti(self.model.parameters(), learning_rate)

    def get_dimension(self):
        """Get the dimensions of the model's weight."""
        return self.model.weight.size().detach().numpy()

    def train(self, epochs=None, batch=False, batch_size=10000):
        """
        Train the model.

        Args:
            epochs (int): Number of training epochs (default is the value set during initialization).
            batch_size (int): Batch size for training data.
        """
        epochs = epochs if epochs is not None else self.epochs
        accuracy=-1
        if batch:
            train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        loss=0
        for self.epochs in range(self.epochs+1, self.epochs + epochs+1):
            if batch:
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device).double()
                    batch_y = batch_y.to(self.device).double()
                    y_predicted = self.model(batch_x)
                    loss = self.criterion(y_predicted, batch_y)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()     
            else:
                y_predicted=self.model(self.dataX)
                loss = self.criterion(y_predicted, self.dataY)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()  
            accuracy=False             

            if (self.epochs) % self.every == 0:
                if self.log:
                    print(f'epoch: {self.epochs}, loss = {loss.item():.4f}')
                for x in self.savers:
                    if ((self.epochs)//self.every)%x[1]==0:
                        if accuracy==False:
                            accuracy = self.accuracy()
                        x[0].save_log(self.epochs,accuracy,loss.item())
            if (self.epochs) % self.saveEvery == 0:
                self.save_model()
        if epochs>0 and len(self.finishers)>0:
            accuracy = self.accuracy()
            for finisher in self.finishers:
                finisher.save_log(self.epochs,accuracy,loss.item())
            
        

    def accuracy(self, operator=False, format=False):
        """
        Compute the accuracy of the model on the test data.

        Args:
            format (bool): Whether to format test data using the set format function.

        Returns:
            float: Accuracy of the model.
        """
        if not hasattr(self, 'dataXTest') or not hasattr(self, 'dataYTest'):
            raise Exception("No default testing data specified. Use set_testing_data(dataXTest, dataYTest).")
        
        if operator==False and self.operator==True:
            X = self.format(self.dataXTest) if format else self.dataXTest
            X = X.to(self.device)
            predicted = self.model(X)
            if self.device == 'cuda':
                predicted = predicted.to("cpu")
            accuracy = torchmetrics.functional.accuracy(predicted, self.dataYTest)
            return accuracy.item()
        i=0
        numberGood=0
        if operator==False:
            operator=self.operator
        X = self.format(self.dataXTest) if format else self.dataXTest
        X = X.to(self.device)
        predicted = self.model(X)
        if self.device=='cuda':
            predicted = predicted.to("cpu")
        dataYTestCPU = self.dataYTest.cpu()
        for i in range(len(X)):
            if operator(predicted[i].detach().numpy(),dataYTestCPU[i]):
                numberGood+=1
        return numberGood/len(X)

    def set_testing_data(self, dataX, dataY,format=True):
        """
        Set the testing data.

        Args:
            dataX (torch.Tensor or numpy.ndarray): Input features for testing.
            dataY (torch.Tensor or numpy.ndarray): Target values for testing.
        """
        if format==True:
            dataX = self.format(dataX)
        self.dataXTest = dataX.to(self.device)
        self.dataYTest = dataY.to(self.device)


    def save_model(self, model_path=True, optimizer_path=None, extra_info=None):
        """
        Save the trained model, optimizer state, and extra information to a file.

        Args:
            model_path (str): Path to the file where the model will be saved.
            optimizer_path (str, optional): Path to the file where the optimizer state will be saved.
            extra_info (dict, optional): Additional information to save alongside the model.
        """
        if model_path==True:
            model_path=f"saves/{self.dataset}_{self.name}_{self.get_dimension()}_{self.poisoning}_{int(self.poisonRate*1000)}.model"
        state = {
            'epoch': self.epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'extra_info': extra_info
        }
        self.save_epochs=self.epochs
        torch.save(state, model_path)
        if optimizer_path:
            torch.save(self.optimizer.state_dict(), optimizer_path)

    def load_model(self, model_path=True, optimizer_path=None):
        """
        Load a trained model, optimizer state, and extra information from a file.

        Args:
            model_path (str): Path to the file from which the model will be loaded.
            optimizer_path (str, optional): Path to the file from which the optimizer state will be loaded.
        """
        try:
            if model_path==True:
                model_path=f"saves/{self.dataset}_{self.name}_{self.get_dimension()}_{self.poisoning}_{int(self.poisonRate*1000)}.model"
            checkpoint = torch.load(model_path)
            self.epochs = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer_path: ### This line should be removed because we should load the optimizer even if we don't have the optimizer path as it is already saved
                self.optimizer.load_state_dict(torch.load(optimizer_path))

            extra_info = checkpoint.get('extra_info', None)
            print("Loaded model from : "+model_path)
            self.save_epochs=self.epochs
            if extra_info:
                print("Loaded extra information:", extra_info)
        except:
            print("Couldn't load the model")
            pass

    def save_logs(self,epochs,accuracy,loss,savers=True):
        if savers==True:
            savers=range(len(self.savers))
        for i in savers:
            self.savers[i][0].save_log(epochs,accuracy,loss)



class Saver:
    def __init__(self):
        self.initiallized=False

    def init(self,name,dimension,dataset,poison,poisonRate):
        self.name=name
        self.dimension=dimension
        self.dataset=dataset
        self.poison=poison
        self.poisonRate=poisonRate
        self.initiallized=True

    def save_log(self,epochs,accuracy,loss):
        print("No save_log function defined for ",self)

