import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch import optim
import math
import numpy as np
import torchvision

class RegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(46,20)
        self.fc2 = nn.Linear(20,20)
        self.fc3 = nn.Linear(20,20)
        self.fc4 = nn.Linear(20,10)
        self.fc5 = nn.Linear(10,1)
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.fc5(x)
        return x

def trainNet(x_train, y_train):
   
    y_train = y_train.view(-1,1)
    train = data_utils.TensorDataset(x_train, y_train)
    batch_size = 1000

    # 80:20 split of train data into train and validation
    train, validation = data_utils.random_split(train,[16960,4240])
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)
    validation_loader = data_utils.DataLoader(validation, batch_size=batch_size, shuffle=True)

    model = RegNet()
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.005)
    epochs = 1000

    # only save model if validation loss decreases
    min_validation_loss = np.inf

    # initialize params for early stopping
    patience = 10
    triggertimes = 0

    for e in range(epochs):
        train_loss = 0.0
        for features, labels in train_loader:
            model.train()
            features = features.view(features.shape[0], -1)
        
            # TODO: Training pass
            optimizer.zero_grad()
            
            output = model.forward(features)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        validation_loss = 0.0
        model.eval()

        for features, labels in validation_loader:
            output = model.forward(features)
            loss = criterion(output, labels)
            validation_loss += loss.item()
        
        print(f'Epoch {e+1} \tTraining Loss:{math.sqrt(train_loss / len(train_loader))} \tValidation Loss: {math.sqrt(validation_loss / len(validation_loader))}')

        if min_validation_loss > validation_loss:
            triggers = 0
            print(f'Validation Loss Decreased({math.sqrt(min_validation_loss/len(validation_loader))}--->{math.sqrt(validation_loss/len(validation_loader))}) \t Saving The Model')
            min_validation_loss = validation_loss
            # Saving State Dict
            torch.save(model.state_dict(), 'saved_model.pth')
        else:
            triggers += 1
            if triggers > patience:
                return
    return 



