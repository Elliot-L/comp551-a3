# not mine, source is the class PyTorch tutorial

from __future__ import print_function

import os, pickle

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 

from torchvision import transforms 
from torch.autograd import Variable 

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
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

def toy_ex( batch_size=64, input_dim=1000, hidden_layer_size=100, output_dim=10, epochs=100 ):
    """
    Launches toy example of a two-layer-deep FC FF Neural Net
    for the purpose of 10-label classification.

    Arguments:

        batch_size: int representing the batch size (number of instances).

        input_dim: int representing the number of features per instance.

        hidden_layer_size: int representing the number of neurons in each hidden layer.

        output_dim: int representing the number of neurons in the output layer
                    (and hence the number of labels).
        
        epochs: int representing the number of learning epochs.

    Returns:

        Nothing.
    """
    x = torch.randn( batch_size, input_dim )
    y = torch.randn( batch_size, output_dim, requires_grad=False )
    model = TwoLayerNet( input_dim, hidden_layer_size, output_dim )    
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    losses = []
    for epoch in range( epochs ):
        preds = model( x )
        loss = criterion( preds, y )
        losses.append( loss )

        print( f"Epoch\t{epoch};\t\t\tLoss:{loss}" )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    toy_ex()