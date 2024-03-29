from __future__ import print_function

import argparse, os, torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MNIST_exNet( nn.Module ):
    """
    2-Conv > 2-Lin FC FF NN with hardcoded dimensions.
    source: https://github.com/pytorch/examples/blob/master/mnist/main.py
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    

class Basic_ConvNet(nn.Module):
    """
    Convolutional neural network (two convolutional layers),
    slightly modified
    source: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py#L35-L56
    """
    def __init__(self, num_classes=10):
        super(Basic_ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d( 1, 16, kernel_size=3, stride=2, padding=1 ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

class Mnist_CNN(nn.Module):
    """
    slighly modified to handle 64x64 images instead of 28x28
    source: https://pytorch.org/tutorials/beginner/nn_tutorial.html#switch-to-cnn
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 64, 64)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 6)
        return xb.view(-1, xb.size(1))

class Basic_NeuralNet(nn.Module):
    """
    Basic FC FF NN. 
    source: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py#L37-L49
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super(Basic_NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class Tut_TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        2-Lin FC FF NN from class tutorial.
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        
        Args:
            - D_in : input dimension of the data
            - H : size of the first hidden layer
            - D_out : size of the output/ second layer
        """
        super(TwoLayerNet, self).__init__() # intialize recursively 
        self.linear1 = torch.nn.Linear(D_in, H) # create a linear layer 
        self.linear2 = torch.nn.Linear(H, D_out) 

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and
        return a tensor of output data. We can use 
        Modules defined in the constructor as well as arbitrary 
        operators on Variables.
        """
        h_relu = self.linear1(x)
        y_pred = self.linear2(h_relu)
        return y_pred