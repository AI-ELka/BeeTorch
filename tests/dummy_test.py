from beetorch import Poison
import torch


result=Poison.init_poison(Poison.LABEL_FLIPPING,0.4,torch.tensor([[0,1]]),torch.tensor([[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,0,0]]))
print(result[1])
