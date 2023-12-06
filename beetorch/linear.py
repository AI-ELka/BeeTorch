import torch
import torchmetrics
from beetorch import Model


class LinearRegressionModel(Model):
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
        super().__init__(dataX,dataY,name,learning_rate,epochs,log,format,device)
        self.model = torch.nn.Linear(self.dataX.size(1), self.dataY.size(1), bias=True).to(self.device).double()
        #self.optimizer = torch.optim.SGD(self.model.parameters(), learning_rate)
        # better to use ADAM optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate) 


    def get_dimension(self):
        """Get the dimensions of the model's weight."""
        temp = list(self.model.weight.size())
        return temp[0]*temp[1]

            
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

