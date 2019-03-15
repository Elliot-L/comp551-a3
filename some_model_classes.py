from __future__ import print_function

import argparse, os, torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class depthwise_separable_conv(nn.Module):
    #  https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843/5
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class pytorch_Tutorial_CNN( nn.Module ):
    """
    source: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    """
    def __init__(self):
        super(pytorch_Tutorial_CNN, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features    

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
        # add a Lin
    def forward(self, xb):
        xb = xb.view(-1, 1, 64, 64)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.max_pool2d(xb, 6)
        return xb.view(-1, xb.size(1))

class Basic_NeuralNet(nn.Module):
    """
    Basic FC FF NN. 
    SC tried this model with 5000 steps, never reached an accuracy > 0.2.
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
        super(Tut_TwoLayerNet, self).__init__() # intialize recursively 
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

class Other_MNIST_CNN(nn.Module):
    # Not mine, source: https://github.com/pytorch/examples/blob/master/mnist/main.py
    def __init__(self):
        super(Other_MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1) 
        self.fc1 = nn.Linear(1024, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # [64, 1, 64, 64] -> [64, 16, 32, 32]
        x = F.max_pool2d(x, 2, 2) # [64, 16, 32, 32] -> [64, 16, 16, 16]
        x = F.relu(self.conv2(x)) # [64, 16, 16, 16] -> [64, 64, 8, 8]
        x = F.max_pool2d(x, 2, 2) # [64, 64, 8, 8] -> [64, 64, 4, 4]
        x = x.view(64, 1024) # [64, 64, 4, 4] -> [64, 1024]
        x = F.relu(self.fc1(x)) # [64, 1024] -> [64, 500]
        x = self.fc2(x) # [64, 500] -> [64, 10]
        return F.log_softmax(x, dim=1)


class Elliot_Model(nn.Module):
    # Not mine, source: https://github.com/pytorch/examples/blob/master/mnist/main.py
    def __init__(self):
        super(Elliot_Model, self).__init__()
        # self.depthwise_separable = depthwise_separable_conv(1, 1)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(128)
        # self.fc1 = nn.Linear(1024, 500)
        # self.fc2 = nn.Linear(500, 10)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # x = F.relu(self.depthwise_separable(x))
        x = F.relu(self.conv1_bn(self.conv1(x))) # [64, 1, 64, 64] -> [64, 16, 32, 32]
        # x = F.relu(self.conv1(x)) # [64, 1, 64, 64] -> [64, 16, 32, 32]
        x = F.max_pool2d(x, 2, 2) # [64, 16, 32, 32] -> [64, 16, 16, 16]
        x = F.relu(self.conv2_bn(self.conv2(x))) # [64, 16, 16, 16] -> [64, 64, 8, 8]
        # x = F.relu(self.conv2(x)) # [64, 16, 16, 16] -> [64, 64, 8, 8]
        x = F.max_pool2d(x, 2, 2) # [64, 64, 8, 8] -> [64, 64, 4, 4]
        x = F.relu(self.conv3_bn(self.conv3(x)))  # [64, 64, 4, 4] -> [64, 128, 2, 2]
        x = x.view(64, 512) # [64, 64, 4, 4] -> [64, 1024]
        x = F.relu(self.fc1(x)) # [64, 1024] -> [64, 500]  # new shape above
        x = self.fc2(x) # [64, 500] -> [64, 10]  # new shape above
        return F.log_softmax(x, dim=1)

class Other_MNIST_SANITY_CHECK_CNN(nn.Module):
    # Not mine, source: https://github.com/pytorch/examples/blob/master/mnist/main.py
    # is a copy of Other_MNIST_CNN but with the correct layer dimensions
    def __init__(self):
        super(Other_MNIST_SANITY_CHECK_CNN, self).__init__()
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
    