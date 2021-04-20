"""
Usage: Coalitional Game, Approximating the contribution of the i-th feature’s value
Python version: 3.9.X
Author: rauls19
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable 
from torch.utils.data import DataLoader

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_epochs, learning_rate, verbose = False):
        super(LSTM, self).__init__()

        self.num_epochs =  num_epochs #1000 epochs
        self.learning_rate = learning_rate #0.001 lr
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.dropout = nn.Dropout(p= 0.05) # Dropout
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True) #lstm

        self.fc_1 =  nn.Linear(hidden_size, hidden_size//2) #fully connected 1
        self.fc = nn.Linear(hidden_size//2, 1) #fully connected last layer
        self.relu = nn.ReLU()

        self.verbose = verbose


    def __generateModel(self):
        self.lstm_model = LSTM(self.input_size, self.hidden_size, self.num_layers, self.num_epochs, self.learning_rate)

    def forward(self,x):
        
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        _, (hn, cn) = self.lstm(x, (h_0.detach(), c_0.detach())) #lstm with input, hidden, and internal state
        hn_fs = hn.view(self.num_layers, x.size(0), self.hidden_size)[-1] #reshaping the data for Dense layer next
        out = self.dropout(hn_fs) # Dropout
        out = self.fc_1(out) # Dense
        out = self.relu(out) # Activation
        out = self.fc(out) # Dense
        out = self.relu(out) # Activation
        return out
    
    def dataPreparation(self, x_train, x_test, y_train):

        self.__generateModel()

        x_train_tensors = Variable(torch.Tensor(x_train.values))
        x_test_tensors = Variable(torch.Tensor(x_test.values))
        y_train_tensors = Variable(torch.Tensor(y_train.values))
        x_train_tensors_final = torch.reshape(x_train_tensors, (x_train_tensors.shape[0], 1, x_train_tensors.shape[1]))
        x_test_tensors_final = torch.reshape(x_test_tensors, (x_test_tensors.shape[0], 1, x_test_tensors.shape[1]))

        x_train_loader = DataLoader(x_train_tensors, batch_size= 200, shuffle= False, num_workers = 0)
        x_test_loader = DataLoader(x_test_tensors, batch_size= 200, shuffle= False, num_workers = 0)
        y_train_loader = DataLoader(y_train_tensors, batch_size= 200, shuffle= False, num_workers = 0)

        return x_train_loader, y_train_loader, x_test_tensors_final

    def fit(self, x_train_loader, y_train_loader):
        
        criterion = torch.nn.MSELoss()    # mean-squared error for regression
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        for epoch in range(self.num_epochs):
            for dtx, dty in zip(enumerate(x_train_loader), enumerate(y_train_loader)):
                xtr = torch.reshape(Variable(dtx[1]), (dtx[1].shape[0], 1, dtx[1].shape[1])) # Reshape
                outputs = self.lstm_model.forward(xtr) #forward pass
                optimizer.zero_grad() #caluclate the gradient, manually setting to 0
                loss = criterion(outputs, dty[1]) # obtain the loss function
                loss.backward() #calculates the loss of the loss function
                optimizer.step() #improve from loss, i.e backprop
            if self.verbose:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
    
    def predict(self, x_test_tensors_final):
        train_predict = self.lstm_model(x_test_tensors_final) # forward pass
        data_predict = train_predict.data.numpy()
        return data_predict