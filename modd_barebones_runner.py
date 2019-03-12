# warning: VSC is weird with torch
# README: modified version of barebones_runner.py to run on our dataset.

from __future__ import print_function

import os, argparse

from tqdm import tqdm
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
import torchvision

from torchvision import transforms
from logger import Logger
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

from tensorboardX import SummaryWriter

# local files
from data_loader import load_training_data, load_training_labels
from some_model_classes import *
from biggest_bbox_extractor import cut_out_dom_bbox

def train(args, model, loss_fn, device, train_loader, validation_loader, optimizer, epoch, minibatch_size, logger):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        # unsqueeze(x) adds a dimension in the xth-position from the left to deal with the Channels argument of the Conv2d layers
        data = data.unsqueeze( 1 )
        data, target = data.to(device), target.to(device)        
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % args.log_interval == 0:
            
            # compute current training accuracy
            with torch.no_grad(): # so to not fuck up our gradients
                preds = model( data )
                preds = preds.argmax( dim=1, keepdim=True ) # get the index of the max log-probability
                correct = preds.eq( target.view_as(preds)).sum().item()
                
                train_acc = 100. * correct / ( minibatch_size )

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}\tTrain Accuracy:{:.1f}%\n'.format(
                    epoch, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.item(), train_acc ) )
                

                val_loss, val_acc = validate(args, model, loss_fn, device, validation_loader, minibatch_size)
                # ================================================================== #
                #                        Tensorboard Logging                         #
                # ================================================================== #

                # 1. Log scalar values (scalar summary)
                info = {    
                    'train loss': loss.item(), 'train accuracy': train_acc, 
                    'val loss': val_loss,  'val accuracy': val_acc
                }
                
                step = batch_idx + ( epoch * ( len( train_loader ) ) )
                print( f">>> Step:{step}" )
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, step)

                # 2. Log values and gradients of the parameters (histogram summary)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.data.cpu().numpy(), step)
                    logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), step)

                # 3. Log training images (image summary)
                '''info = { 'images': images.view(-1, 64, 64)[:10].cpu().numpy() }

                for tag, images in info.items():
                    logger.image_summary(tag, images, step)'''

def validate(args, model, loss_fn, device, validation_loader, minibatch_size):
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in validation_loader:
            data = data.unsqueeze( 1 )
            data, target = data.to(device), target.to(device)
            output = model(data)
            validation_loss += loss_fn(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    validation_loss /= len(validation_loader)
    accuracy = 100. * correct / ( len(train_loader) * minibatch_size )

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        validation_loss, correct, ( len(train_loader) * minibatch_size ),
        accuracy))

    return validation_loss, accuracy

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
        
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    # Dataset loading
    print( "\n>>> Loading datasets\n" )
    training_data_raw = load_training_data( 'train_images.pkl', as_tensor=False ) 
    cleaned_images = [ cut_out_dom_bbox( training_data_raw[i,:,:] )[0] for i in range( training_data_raw.shape[0] ) ]
    training_data = torch.stack( cleaned_images )
    training_labels = load_training_labels( 'train_labels.csv', as_tensor=True )
    tensor_dataset = TensorDataset( training_data, training_labels.long() )
    assert len( training_labels ) == len( training_data )

    # Creating train/validation datasets
    print( "\n>>> Splitting datasets\n" )
    dataset_size = len( training_data )
    indices = list( range( dataset_size ) )
    validation_split = 0.2 # could be parameterized
    split = int( np.floor( validation_split * dataset_size ) )

    try:
        assert ( len( training_data )*( 1.0 - validation_split )  )%args.batch_size == 0
    except AssertionError as err:
        raise Exception( f"Error: your chosen test/val split ({1.0-validation_split} , {validation_split}) doesn't match your minibatch size ({args.minibatch_size}).\nMake sure that both your train and validation datasets' length is a multiple of your args.minibatch_size argument.") from err

    np.random.seed( args.seed )
    np.random.shuffle( indices )
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler( train_indices )
    valid_sampler = SubsetRandomSampler( val_indices )

    train_loader = torch.utils.data.DataLoader(
        tensor_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler
        # shuffle=True # already shuffled
    )
    validation_loader = torch.utils.data.DataLoader(
        tensor_dataset, 
        batch_size=args.batch_size,
        sampler=valid_sampler
        # shuffle=True # already shuffled
    )

    start_timestamp = datetime.now().strftime( '%Y-%m-%d_%H-%M' )

    logpath = os.path.join( os.getcwd(), 'logs', start_timestamp )

    if not os.path.isdir( logpath ):
        os.mkdir( logpath )
    
    logger = Logger( logpath )

    # Model definition
    model = Other_MNIST_CNN().to( device ).double() # casting it to double because of some pytorch expected type peculiarities

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  
    loss_fn = nn.CrossEntropyLoss() 

    print( "\n>>> Starting training\n" )

    for epoch in range( args.epochs ):
        train(args, model, loss_fn, device, train_loader, validation_loader, optimizer, epoch, args.batch_size, logger)
        #test(args, model, loss_fn, device, validation_loader) # was test_loader

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        with open( os.path.join( os.getcwd(), 'pickled-params', datetime.now().strftime( '%Y-%m-%d_%H-%M' ) ), 'w' ) as params_file:
            params_file.write( args.__repr__() )

