# warning: VSC is weird with torch
# README: modified version of barebones_runner.py to run on our dataset.

from __future__ import print_function

import os

import numpy as np

import torch
import torch.nn as nn
import torchvision

from torchvision import transforms
from logger import Logger
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from tensorboardX import SummaryWriter

# local files
from data_loader import load_training_data, load_training_labels
from some_model_classes import *
from biggest_bbox_extractor import cut_out_dom_bbox

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset loading
training_data_raw = load_training_data( 'train_images.pkl', as_tensor=False ) 
cleaned_images = [ cut_out_dom_bbox( training_data_raw[i,:,:] )[0] for i in range( training_data_raw.shape[0] ) ]
training_data = torch.stack( cleaned_images )
training_labels = load_training_labels( 'train_labels.csv', as_tensor=True )
tensor_dataset = TensorDataset( training_data, training_labels.long() )

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=tensor_dataset, 
                                          batch_size=100, 
                                          shuffle=True)

model = Mnist_CNN().to(device).double() # casting it to double because of some pytorch expected type peculiarities

logger = Logger('./logs')

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)  

data_iter = iter(data_loader)
iter_per_epoch = len(data_loader)
total_step = 5000

# Start training
for step in range(total_step):
    
    # Reset the data_iter
    if (step+1) % iter_per_epoch == 0:
        data_iter = iter(data_loader)

    # Fetch images and labels
    images, labels = next(data_iter)
    images, labels = images.view(images.size(0), -1).to(device), labels.to(device)
    
    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Compute accuracy
    _, argmax = torch.max(outputs, 1)
    accuracy = (labels == argmax.squeeze()).float().mean()

    if (step+1) % 100 == 0:
        print ('Step [{}/{}], Loss: {:.4f}, Acc: {:.2f}' 
               .format(step+1, total_step, loss.item(), accuracy.item()))

        # ================================================================== #
        #                        Tensorboard Logging                         #
        # ================================================================== #

        # 1. Log scalar values (scalar summary)
        info = { 'loss': loss.item(), 'accuracy': accuracy.item() }

        for tag, value in info.items():
            logger.scalar_summary(tag, value, step+1)

        # 2. Log values and gradients of the parameters (histogram summary)
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, value.data.cpu().numpy(), step+1)
            logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), step+1)

        # 3. Log training images (image summary)
        info = { 'images': images.view(-1, 64, 64)[:10].cpu().numpy() }

        for tag, images in info.items():
            logger.image_summary(tag, images, step+1)