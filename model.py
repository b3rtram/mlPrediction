import torch
import torch.nn as nn
from numpy import array
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.c1 = torch.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=10, stride=1)
        self.m1 = torch.nn.MaxPool1d(kernel_size=2)
        self.c2 = torch.nn.Conv1d(32, 64, kernel_size=3)
        self.r2 = torch.nn.ReLU()
        self.m2 = torch.nn.MaxPool1d(kernel_size=2)
        self.f = torch.nn.Flatten()
        self.d = torch.nn.Dropout(p=0.25)
        self.l1 = torch.nn.Linear(448, 1)
        self.learning_rate = 0.01
        self.epochs = 10

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(list(self.parameters()), lr=self.learning_rate)
        
    def forward(self, x):
        x = self.c1(x)
        x = self.r2(x)
        x = self.m1(x)
        x = self.c2(x)
        x = self.r2(x)
        x = self.m2(x)
        x = self.f(x)
        x = self.d(x)
        x = self.l1(x)
        return x

    def modelTrain(self, values):
        dataloader = self.generateTrain(values)
        self.train()

        for i in range(self.epochs):
            train_loss = 0.0

            for data, target in dataloader:
                self.optimizer.zero_grad()
                data = data.view(data.shape[0], 1, -1)
                output = self(data)
                loss = self.loss_fn(output, target)
                train_loss += loss.item()

                loss.backward()

                self.optimizer.step()

        print(train_loss)

    def generateTrain(self, values):
        w = 42
        step = 1
        batch_size = 1

        x, y = [], []
        
        for i in range(len(values)):
            start = i + w
            end = start + step

            if end > len(values):
                break

            buf_x = values[i:start]  
            buf_y = values[start:end]
            
            x.append(buf_x)
            y.append(buf_y)

        x = array(x)
        y = array(y)


        tensor_x = torch.FloatTensor(x)
        tensor_y = torch.FloatTensor(y)

        print(tensor_x.shape)
        print(tensor_y.shape)

        train_dataset = TensorDataset(tensor_x, tensor_y)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)

        return dataloader
    