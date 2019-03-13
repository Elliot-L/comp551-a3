# warning: VSC is weird with torch
# README: modified version of barebones_runner.py to run on our dataset.

from __future__ import print_function

import os, argparse

from tqdm import tqdm
from datetime import datetime
from collections import Counter

import numpy as np

import torch
import torch.nn as nn
import torchvision

from torchvision import datasets, transforms
from logger import Logger
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
                
                train_acc = 100. * correct / ( train_loader.batch_size )

                print('Training Epoch: {} [{}/{} ({:.0f}%)]\t\tTrain Loss: {:.6f}\tTrain Accuracy:{:.1f}%\n'.format(
                    epoch, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.item(), train_acc ) )
                

                # ================================================================== #
                #                        Tensorboard Logging                         #
                # ================================================================== #

                # 1. Log scalar values (scalar summary)
                info = {    
                    'train loss': loss.item(), 'train accuracy': train_acc
                }
                
                step = batch_idx + ( epoch *  len( train_loader ) ) 
                print( f">>> Step:{step}\n" )
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, step)

                # 2. Log values and gradients of the parameters (histogram summary)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.data.cpu().numpy(), step)
                    logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), step)

                # 3. Log training images (image summary), commented out manually
                '''info = { 'images': images.view(-1, 64, 64)[:10].cpu().numpy() }

                for tag, images in info.items():
                    logger.image_summary(tag, images, step)'''

def validate(args, model, loss_fn, device, validation_loader, epoch, logger, validation_split_fraction ):
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

    validation_loss /= ( validation_loader.batch_size * len( validation_loader ) )
    accuracy = 100. * correct / ( validation_loader.batch_size * len( validation_loader ) )

    print('\nValidation set:\t\tAverage loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        validation_loss, correct, ( validation_loader.batch_size * len( validation_loader ) ),
        accuracy))

    # ================================================================== #
    #                        Tensorboard Logging                         #
    # ================================================================== #
    # 1. Log scalar values (scalar summary)
    info = { 'val loss': validation_loss, 'val accuracy': accuracy }
    
    step = epoch * ( len( validation_loader.dataset ) * ( 1. - validation_split_fraction ) ) / validation_loader.batch_size
    for tag, value in info.items():
        logger.scalar_summary(tag, value, step)
    
    # 2. Log values and gradients of the parameters (histogram summary)
    for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.data.cpu().numpy(), step)
        logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), step)

    # 3. Log training images (image summary), commented out manually
    '''info = { 'images': images.view(-1, 64, 64)[:10].cpu().numpy() }

    for tag, images in info.items():
        logger.image_summary(tag, images, step)'''

def sanity_check_train(args, model, device, train_loader, optimizer, epoch, train_fraction, logger):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
        
            # compute current training accuracy:
            with torch.no_grad():
                preds = model( data )
                preds = preds.argmax( dim=1, keepdim=True ) # get the index of the max log-probability
                correct = preds.eq( target.view_as(preds)).sum().item()

                train_acc = 100. * correct / ( train_loader.batch_size )

                print('<Sanity-checking on Standard MNIST> Training Epoch: {} [{}/{} ({:.0f}%)]\t\tTrain (Total) Loss: {:.6f},\tTrain (Total) Accuracy:{:.1f}%'.format(
                    epoch, batch_idx * len(data), train_fraction*len(train_loader.dataset),
                    100. * batch_idx * len(data) / ( train_fraction * len(train_loader.dataset) ), loss.item(), train_acc))


                # ================================================================== #
                #                        Tensorboard Logging                         #
                # ================================================================== #
                # 1. Log scalar values (scalar summary)
                info = { 'train loss': loss.item(), 'train accuracy': train_acc }
                
                step = batch_idx + ( epoch * len( train_loader ) )
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, step)
                
                # 2. Log values and gradients of the parameters (histogram summary)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.data.cpu().numpy(), step)
                    logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), step)

                # 3. Log training images (image summary), commented out manually
                '''info = { 'images': images.view(-1, 64, 64)[:10].cpu().numpy() }

                for tag, images in info.items():
                    logger.image_summary(tag, images, step)'''

def sanity_check_validate(args, model, device, validation_loader, epoch, logger, validation_split_fraction ):
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            validation_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    validation_loss /= ( validation_loader.batch_size * len( validation_loader ) )
    accuracy = 100. * correct / ( validation_loader.batch_size * len( validation_loader) )
    print('\n<Sanity-checking on Standard MNIST> Validating Epoch: {}\t\t\t\tValidating (Mean) Loss: {:.4f},\tValidating (Mean) Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, validation_loss, correct, validation_loader.batch_size * len(validation_loader),
        #100. * correct / len(validation_loader.dataset)))
        accuracy))

    # ================================================================== #
    #                        Tensorboard Logging                         #
    # ================================================================== #
    # 1. Log scalar values (scalar summary)
    info = { 'val loss': validation_loss, 'val accuracy': accuracy }
    
    step = epoch * ( len( validation_loader.dataset ) * ( 1. - validation_split_fraction ) ) / validation_loader.batch_size
    for tag, value in info.items():
        logger.scalar_summary(tag, value, step)
    
    # 2. Log values and gradients of the parameters (histogram summary)
    for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.data.cpu().numpy(), step)
        logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), step)

    # 3. Log training images (image summary), commented out manually
    '''info = { 'images': images.view(-1, 64, 64)[:10].cpu().numpy() }

    for tag, images in info.items():
        logger.image_summary(tag, images, step)'''


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--validation-split-fraction', type=float, default=0.2, metavar='V',
                        help='the fraction (0.#) of the training dataset to set aside for validation')
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
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--MNIST-sanity-check', type=bool, default=False,
                        help="Whether to run the model on PyTorch's MNIST dataset first")
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--verbose', type=bool, default=True,
                        help='boolean indicator of verbosity')
    args = parser.parse_args()
        
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset loading
    print( "\n>>> Loading datasets\n" )
    tensor_dataset = None # dummy declaration
    if args.MNIST_sanity_check == True:
        tensor_dataset = datasets.MNIST(
            os.path.join( '..', 'data' ), 
            train=True, 
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]
            )
        )
        if args.verbose:
            print( f">>> MNIST dataset shape = {tensor_dataset.data.shape}" )

    else:
        training_data_raw = load_training_data( 'train_images.pkl', as_tensor=False ) 
        cleaned_images = [ cut_out_dom_bbox( training_data_raw[i,:,:] )[0] for i in range( training_data_raw.shape[0] ) ]
        training_data = torch.stack( cleaned_images )
        if args.verbose:
            print( ">>> Loaded and cleaned (extracted) training data" )
        training_labels = load_training_labels( 'train_labels.csv', as_tensor=True )
        tensor_dataset = TensorDataset( training_data, training_labels.long() )
        assert len( training_labels ) == len( training_data )
        if args.verbose:
            print( ">>> Compiled tensor dataset" )

    # Creating train/validation datasets
    print( "\n>>> Splitting datasets\n" )
    dataset_size = len( tensor_dataset )
    indices = list( range( dataset_size ) )
    split = int( np.floor( args.validation_split_fraction * dataset_size ) )
    if args.verbose:
        print( f">>> Dataset size: {dataset_size}" )
        print( f">>> Validation split: {args.validation_split_fraction}" )
        print( f">>> Number of validation instances: {split}" )

    if args.verbose:
        print( f">>> Randomizing dataset prior to splitting" )

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

    if args.verbose:
        print( ">>> Split original dataset into train/validate datasets" )
        print( f">>> Training dataset was built from a dataset of {len( train_loader.dataset )} instances" )
        print( f"... and made into {len( train_loader )} minibatches of {train_loader.batch_size} instances" )
        print( f"... ( {len( train_loader )} * {train_loader.batch_size} = {len( train_loader ) * train_loader.batch_size} )\n" )
        print( f">>> Validating dataset was built from a dataset of {len( validation_loader.dataset )} instances" )
        print( f"... and made into {len( validation_loader )} minibatches of {validation_loader.batch_size} instances" )
        print( f"... ( {len( validation_loader)} * {validation_loader.batch_size} = {len( validation_loader ) * validation_loader.batch_size} )\n" )

        # making sure the datasets are balanced wrt labels
        training_labels, validating_labels = [], [] 
        with torch.no_grad():
            for _, target in train_loader:
                training_labels.extend( [ t.item() for t in  target ] )
            for _, target in validation_loader: 
                validating_labels.extend( [ t.item() for t in  target ] )
        
        training_labels_counter = Counter( training_labels )
        for k,v in training_labels_counter.items():
            training_labels_counter[ k ] = '{:.3f}%'.format( 100.* v / len( training_labels ) )

        validating_labels_counter = Counter( validating_labels )
        for k,v in validating_labels_counter.items():
            validating_labels_counter[ k ] = '{:.3f}%'.format( 100.* v / len( validating_labels ) )

        del training_labels # for the sake of memory management
        del validating_labels # for the sake of memory management
        print( ">>> The label distribution in the training dataset:\n{}\n".format( '\n'.join( [ ( k,v ).__str__() for k,v in training_labels_counter.items() ] ) ) )
        print( ">>> The label distribution in the validating dataset:\n{}\n".format( '\n'.join( [ ( k,v ).__str__() for k,v in validating_labels_counter.items() ] ) ) )

    try:
        assert ( len( train_loader.dataset ) * ( 1.0 - args.validation_split_fraction )  ) % train_loader.batch_size == 0
        assert ( len( validation_loader.dataset ) * args.validation_split_fraction ) % validation_loader.batch_size == 0
    except AssertionError as err:
        raise Exception( f"Error: your chosen test/val split ({1.0-args.validation_split_fraction} , {args.validation_split_fraction}) doesn't match your minibatch size ({args.batch_size}).\nMake sure that both your train and validation datasets' length is a multiple of your args.batch-size argument.") from err

    start_timestamp = datetime.now().strftime( '%Y-%m-%d_%H-%M' )

    logpath = os.path.join( os.getcwd(), 'logs', start_timestamp )
    
    if args.MNIST_sanity_check:
        logpath = os.path.join( os.getcwd(), 'logs', 'MNIST-Sanity-Check'+start_timestamp )
    
    if not os.path.isdir( logpath ):
        os.mkdir( logpath )
    
    logger = Logger( logpath )
    if args.verbose:
        print( f"\nThe log file will be saved in {logpath.__str__()}\n")

    # Model definition
    model = Other_MNIST_CNN().to( device ).double() # casting it to double because of some pytorch expected type peculiarities

    if args.MNIST_sanity_check == True:
        model = Other_MNIST_SANITY_CHECK_CNN().to( device ) # _not_ casting it to double because of some pytorch expected type peculiarities

    # Loss and optimizer
    optimizer = torch.optim.Adam( model.parameters(), lr=args.lr )  
    loss_fn = nn.CrossEntropyLoss() 

    print( "\n>>> Starting training\n" )

    for epoch in range( args.epochs ):
        if args.MNIST_sanity_check == True:
            sanity_check_train( args, model, device, train_loader, optimizer, epoch, ( 1.0 - args.validation_split_fraction ), logger )
            sanity_check_validate( args, model, device, validation_loader, epoch, logger, args.validation_split_fraction )
        else:
            train(args, model, loss_fn, device, train_loader, validation_loader, optimizer, epoch, args.batch_size, logger)
            validate(args, model, loss_fn, device, validation_loader, epoch, logger, args.validation_split_fraction )
    
    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        with open( os.path.join( os.getcwd(), 'pickled-params', datetime.now().strftime( '%Y-%m-%d_%H-%M' ), '.pickle' ), 'w' ) as params_file:
            params_file.write( args.__repr__() )
            params_file.write( '\n' )
            params_file.write( optimizer.__repr__() )
            params_file.write( '\n' )
            params_file.write( loss_fn.__repr__() )

    print( f"\nThe log file was saved in {logpath.__str__()}\n")
