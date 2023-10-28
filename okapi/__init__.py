import torch
import numpy as np
import torchmetrics
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


class RegressionModel:
    def __init__(self, dataX, dataY,name , learning_rate=0.01, epochs=400, log=False, format=lambda x: torch.tensor(x).double(), device=None):
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
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataX = format(dataX).to(self.device)
        self.dataY = torch.tensor(dataY).to(self.device)
        self.train_data = TensorDataset(self.dataX, self.dataY)
        self.model = torch.nn.Linear(self.dataX.size(1), self.dataY.size(1), bias=True).to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.SGD(self.model.parameters(), learning_rate)
        self.epochs = epochs
        self.log = log
        self.every = 100
        self.everySave = 500
        self.format = format
        self.dataset="MNIST"

    def set_log(self, log):
        """Set whether to log training progress."""
        self.log = log
    
    def set_dataset(self,dataS):
        self.dataset=dataS

    def add_saver(self,saver):
        saver.init(self.name,len(self.dataX[0]),self.dataset)
        self.savers.append(saver)

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

    def train(self, epochs=None, batch_size=32):
        """
        Train the model.

        Args:
            epochs (int): Number of training epochs (default is the value set during initialization).
            batch_size (int): Batch size for training data.
        """
        epochs = epochs if epochs is not None else self.epochs
        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        for self.epochs in range(self.epochs, self.epochs + epochs):
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                y_predicted = self.model(batch_x)
                loss = self.criterion(y_predicted, batch_y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            if (self.epochs + 1) % self.everySave==0 and self.save:
                try:
                    acc = self.accuracy()
                    self.save_logs(self.epochs,acc,loss.item())
                except:
                    print("Couldn't save accuracy")
                    

            if (self.epochs + 1) % self.every == 0 and self.log:
                print(f'epoch: {self.epochs + 1}, loss = {loss.item():.4f}')
            
        

    def accuracy(self, format=False):
        """
        Compute the accuracy of the model on the test data.

        Args:
            format (bool): Whether to format test data using the set format function.

        Returns:
            float: Accuracy of the model.
        """
        if not hasattr(self, 'dataXTest') or not hasattr(self, 'dataYTest'):
            raise Exception("No default testing data specified. Use set_testing_data(dataXTest, dataYTest).")

        X = self.format(self.dataXTest) if format else self.dataXTest
        X = X.to(self.device)
        predicted = self.model(X)
        if self.device == 'cuda':
            predicted = predicted.to("cpu")
        accuracy = torchmetrics.functional.accuracy(predicted, self.dataYTest)
        return accuracy.item()

    def set_testing_data(self, dataX, dataY):
        """
        Set the testing data.

        Args:
            dataX (torch.Tensor or numpy.ndarray): Input features for testing.
            dataY (torch.Tensor or numpy.ndarray): Target values for testing.
        """
        self.dataXTest = torch.tensor(dataX).to(self.device)
        self.dataYTest = torch.tensor(dataY).to(self.device)


    def save_model(self, model_path=True, optimizer_path=None, extra_info=None):
        """
        Save the trained model, optimizer state, and extra information to a file.

        Args:
            model_path (str): Path to the file where the model will be saved.
            optimizer_path (str, optional): Path to the file where the optimizer state will be saved.
            extra_info (dict, optional): Additional information to save alongside the model.
        """
        if model_path==True:
            model_path="saves/"+self.name+".model"
        state = {
            'epoch': self.epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'extra_info': extra_info
        }
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
        if model_path==True:
            model_path="saves/"+self.name+".model"
        checkpoint = torch.load(model_path)
        self.epochs = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer_path:
            self.optimizer.load_state_dict(torch.load(optimizer_path))

        extra_info = checkpoint.get('extra_info', None)
        if extra_info:
            print("Loaded extra information:", extra_info)

    def save_logs(self,epochs,accuracy,loss,savers=True):
        if savers==True:
            savers=range(len(self.savers))
        for i in savers:
            self.savers[i].save_log(epochs,accuracy,loss)

            
if __name__ == "__main__":
    # Example usage:
    # Create a RegressionModel instance, set data, and train the model.
    dataX_train = torch.rand((100, 1))  # Example training data
    dataY_train = 2 * dataX_train + 1  # Example training labels
    model = RegressionModel(dataX_train, dataY_train)
    model.train(epochs=200)
    
    # Set testing data and compute accuracy
    dataX_test = torch.rand((50, 1))  # Example testing data
    dataY_test = 2 * dataX_test + 1  # Example testing labels
    model.set_testing_data(dataX_test, dataY_test)
    test_accuracy = model.accuracy()
    print(f"Test accuracy: {test_accuracy}")

    # Save and load the model
    model.save_model("my_model.pth")
    loaded_model = RegressionModel(dataX_train, dataY_train)
    loaded_model.load_model("my_model.pth")
    loaded_model.set_testing_data(dataX_test, dataY_test)
    test_accuracy_loaded = loaded_model.accuracy()
    print(f"Test accuracy of loaded model: {test_accuracy_loaded}")

