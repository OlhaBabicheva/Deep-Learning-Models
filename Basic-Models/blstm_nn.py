import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')

# Hyperparameters
in_size = 28 # Nx1x28x28, N <- batch_size
h_size = 256
sequence_len = 28
num_layers = 5
num_classes = 10
l_rate = 1e-3
batch_size = 64
num_epochs = 3

# Load MNIST dataset
train_dataset = datasets.MNIST(root = 'dataset/', train = True, transform = transforms.ToTensor(), download = True)
test_dataset = datasets.MNIST(root = 'dataset/', train = False, transform = transforms.ToTensor(), download = True)

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

# Network
class BLSTM(nn.Module):
    def __init__(self, in_size, h_size, num_layers, num_classes):
        super(BLSTM, self).__init__()
        self.h_size = h_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(in_size, h_size, num_layers, batch_first=True, bidirectional = True)
        self.fc = nn.Linear(h_size*2, num_classes)
    
    def forward(self, x):
        h_0 = torch.zeros(self.num_layers*2, x.size(0), self.h_size).to(device) # first hidden state
        c_0 = torch.zeros(self.num_layers*2, x.size(0), self.h_size).to(device) # first cell state

        x, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        x = self.fc(x[:, -1, :]) # we are sending the last h. state to linear layer
        return x

        

model = BLSTM(in_size, h_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss() # loss function
optimizer = optim.Adam(model.parameters(), lr = l_rate) # optimizer

def train(dataloader, model):
    size = len(dataloader.dataset)
    model.train()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}\n-------------------------------')
        for batch, (X, y) in enumerate(train_loader):
            X = X.to(device).squeeze(1)
            y = y.to(device)

            # Compute predictions and loss
            pred = model(X) # 64x10
            loss = criterion(pred, y)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Accuracy on training data")
    else:
        print("Accuracy on test data")
    model.eval()
    n_correct = 0
    n_samples = 0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device).squeeze(1)
            y = y.to(device)

            pred = model(X)
            _, prediction = pred.max(1) # idx of max value for the second dim
            n_correct += (prediction == y).sum()
            n_samples += prediction.size(0)
        print(f'Accuracy: {n_correct/n_samples * 100:.3f}%')

train(train_loader, model)
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

torch.save(model.state_dict(), 'BLSTM.pth')
