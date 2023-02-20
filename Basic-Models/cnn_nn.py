import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')

# Parameters
in_channels = 1
num_classes = 10
l_rate = 1e-3
batch_size = 64
num_epochs = 10

# Load MNIST dataset
train_dataset = datasets.MNIST(root = 'dataset/', train = True, transform = transforms.ToTensor(), download = True)
test_dataset = datasets.MNIST(root = 'dataset/', train = False, transform = transforms.ToTensor(), download = True)

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

# Network
class CNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size = (3, 3))
        self.conv2 = nn.Conv2d(16, 16, kernel_size = (3, 3))
        self.fc = nn.Linear(16*5*5, num_classes)
        self.pool = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

model = CNN(in_channels, num_classes).to(device)
criterion = nn.CrossEntropyLoss() # loss function
optimizer = optim.Adam(model.parameters(), lr = l_rate) # optimizer

def train(dataloader, model):
    size = len(dataloader.dataset)
    model.train()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}\n-------------------------------')
        for batch, (X, y) in enumerate(train_loader):
            X = X.to(device)
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
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            _, prediction = pred.max(1) # idx of max value for the second dim
            n_correct += (prediction == y).sum()
            n_samples += prediction.size(0)
        print(f'Accuracy: {n_correct/n_samples * 100:.3f}%')

train(train_loader, model)
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

torch.save(model.state_dict(), 'CNN.pth')
