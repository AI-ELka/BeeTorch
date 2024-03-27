import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1: Implement the model
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.out = nn.Linear(hidden_dim3, output_dim)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))  # Sigmoid changed to ReLU as it's more common
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x

# 2: Training the model
def train_model(model, trainloader, criterion, optimizer, epochs=10):
    model.train()  # Set model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)  # Move data to CUDA device
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss/len(trainloader)}")
    print('Finished Training')

# 3: Testing the model
def test_model(model, testloader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)  # Move data to CUDA device
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}')
    return 100 * correct / total



# 4: Label_flipping attack
def label_flipping(dataY, poison_rate):
    if poison_rate==0:
        return dataY
    num_instances_to_flip = int(len(dataY) * poison_rate)
    T = np.random.choice(len(dataY), size=num_instances_to_flip, replace=False)
    print(f"Poisoning with Label Flipping at a rate of {poison_rate}:")
    tmp = dataY[T[-1]]
    tmp1 = 0
    for i in range(num_instances_to_flip):
        tmp1 = dataY[T[i]]
        dataY[T[i]] = tmp
        tmp = tmp1
        tmp1 = 0

    return dataY

# 5: Import dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 6: attack on the dataset
poison_rate = float(input("Type in the poison rate : "))  # Set the poison rate
attack_data = label_flipping(trainset.targets.numpy(), poison_rate)
trainset.targets = torch.tensor(attack_data)

# 7: Preparing dataset
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

epochs = int(input("Type in the number of epochs : "))

# 8: Experimenting with different dimensions of the model
input_dim = 28*28
output_dim = 10
hidden_dim1 = map(int,input('Type in the first hidden dimension : ').strip('][').split(','))
hidden_dim2 = map(int,input('Type in the second hidden dimension : ').strip('][').split(','))
hidden_dim3 = map(int,input('Type in the third hidden dimension : ').strip('][').split(','))
accuracies = []
#number_parametre = []
f = open("outputs/nn_with_label_flipping.txt", "a+")
f.write(f"\n label-flipping  {poison_rate}")
f.write(f"\nhidden_dim1, hidden_dim1, hidden_dim1= {hidden_dim1}")
for h1, h2, h3 in zip(hidden_dim1, hidden_dim2, hidden_dim3):
    print(f"(hidden_dim1, hidden_dim1, hidden_dim1)=({h1},{h2},{h3})")
    f.write(f"\n(hidden_dim1, hidden_dim1, hidden_dim1)=({h1},{h2},{h3})")
    net = Net(input_dim, h1, h2, h3, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    print("training...")
    train_model(net, trainloader, criterion, optimizer, epochs=epochs)
    print("testing....")
    accuracy = test_model(net, testloader)
    accuracies.append(accuracy)

    f.write(f"\nResult for (hidden_dim1, hidden_dim1, hidden_dim1)=({h1},{h2},{h3})")
    f.write(f"\taccuracies = {accuracy}")

#f.write(f"\nnumber_parametre = {number_parametre}")
#f.write(f"\naccuracies = {accuracies}")