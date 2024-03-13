from torch.utils.data import DataLoader, TensorDataset
import torch
from geom_median.torch import compute_geometric_median
from beetorch.linear.linear import LinearRegressionModel
from beetorch import PoisonClass




class RobustPoisonClass(PoisonClass):
    def delegateTraining(self,poison):
        return True
    def training_poison(self,poison,poisonRate,dataX,dataY,criterion,optimizer,model):
        bacth_size = min(int(len(dataY)*poisonRate),500)
        if bacth_size==0:
            bacth_size = 500
        max_poisonned_batch = int(len(dataY)*poisonRate)/bacth_size
        print(bacth_size,max_poisonned_batch)
        LosseGradsBias=[]
        LosseGradsWeight=[]
        n=len(dataY)
        
        for i in range(n//bacth_size):
            if poison==self.CLASSIC_GRADIENT_ATTACK and i<max_poisonned_batch:
                dataP = dataX[i*bacth_size:min(i*bacth_size+bacth_size,int(len(dataY)*poisonRate))]
                print(len(dataP))
                y_predictedP=model(dataP)
                dataG = dataX[max(i*bacth_size,int(len(dataY)*poisonRate)):i*bacth_size+bacth_size]
                print(len(dataG))
                y_predictedG=model(dataG)
                lossP = 0*len(y_predictedP)/bacth_size*criterion(y_predictedP, dataY[i*bacth_size:min(i*bacth_size+bacth_size,int(len(dataY)*poisonRate))])
                lossG = len(y_predictedG)/bacth_size*criterion(y_predictedG, dataY[max(i*bacth_size,int(len(dataY)*poisonRate)):i*bacth_size+bacth_size])
                loss = lossP + lossG
                loss.backward()
            else:
                y_predicted=model(dataX[i*bacth_size:i*bacth_size+bacth_size])
                loss = criterion(y_predicted, dataY[i*bacth_size:i*bacth_size+bacth_size])
                loss.backward()
            for name, param in model.named_parameters():
                if name=='weight':
                    LosseGradsWeight.append(param.grad)
                elif name=='bias':
                    LosseGradsBias.append(param.grad)
            optimizer.zero_grad()
        weights = torch.ones(len(LosseGradsWeight))
        LossWeightMedian = compute_geometric_median(LosseGradsWeight, weights)
        LossBiasMedian = compute_geometric_median(LosseGradsBias, weights)
        for name, param in model.named_parameters():
                if name=='weight':
                    param.grad = LossWeightMedian.median
                elif name=='bias':
                    param.grad = LossBiasMedian.median
        optimizer.step()
        optimizer.zero_grad()
        return loss

class RobustLinearRegressionModel(LinearRegressionModel):
    def __init__(self, dataX, dataY,name , learning_rate=0.01, epochs=0, log=False, format=lambda x: torch.tensor(x).double(), device=None):
        super().__init__(dataX, dataY,name , learning_rate, epochs, log, format, device)
        self.Poison = RobustPoisonClass()
    def train(self, epochs=None, batch=False, batch_size=10000):
        """
        Train the model.

        Args:
            epochs (int): Number of training epochs (default is the value set during initialization).
            batch_size (int): Batch size for training data.
        """
        epochs = epochs if epochs is not None else self.epochs
        if(self.dataX.dtype!=self.dataY.dtype):
            print("Error: dataX and dataY data types don't match")
            return 0
        accuracy=-1
        if batch:
            train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        loss=0
        for self.epochs in range(self.epochs+1, self.epochs + epochs+1):
            if batch:
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    y_predicted = self.model(batch_x)
                    loss = self.criterion(y_predicted, batch_y)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()     
            else:
                if self.Poison.delegateTraining(self.poisoning):
                    loss = self.Poison.training_poison(self.poisoning,self.poisonRate,self.dataX,self.dataY,self.criterion,self.optimizer,self.model)
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