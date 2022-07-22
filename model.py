import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch import optim
import math

#read on transforms.normalize
class RegNetOne(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(46,16)
        self.fc2 = nn.Linear(16,1)

        self.dropout = nn.Dropout(p=0.2)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class RegNetThree(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(46,24)
        self.fc2 = nn.Linear(24,16)
        self.fc3 = nn.Linear(16,8)
        self.fc4 = nn.Linear(8,1)

        self.dropout = nn.Dropout(p=0.2)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

def trainNet(x_train, y_train):
    y_train = y_train.view(-1,1)
    print(f"x: {x_train.shape} y: {y_train.shape}")
    train = data_utils.TensorDataset(x_train, y_train)
    
    train_loader = data_utils.DataLoader(train, batch_size=100, shuffle=True)

    model = RegNetThree()

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.005)
    epochs = 500

    for e in range(epochs):
        running_loss = 0
        for features, labels in train_loader:
            # Flatten MNIST images into a 784 long vector
            features = features.view(features.shape[0], -1)
        
            # TODO: Training pass
            optimizer.zero_grad()
            
            output = model.forward(features)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        else:
            print(f"Training loss: {math.sqrt(running_loss/len(train_loader))}")

    model.eval()
    return model



