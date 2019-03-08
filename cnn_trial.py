# warning: VSC is weird with torch

from __future__ import print_function

import argparse, random, torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from data_loader import load_training_data, load_training_labels
from some_model_classes import *

def shuffle_data_and_labels( data_tensor:torch.tensor, labels_tensor:torch.tensor, seed ):
    """
    Superseded by DataLoader's shuffle parameter.
    Shuffles the input data and label tensors while maintaining their row-wise pair.

    Arguments:

        data_tensor, labels_tensors:    torch tensors of the data and corresponding labels, 
                                        their first dimension must match.
    
        seed: int seed to provide to random.seed.

    Returns:

        training_data, training_labels: torch tensors which have been shuffled while maintaining
                                        the association between the ith instance in data_tensor
                                        and labels_tensor.
    """
    try:
        assert data_tensor.shape[0] == labels_tensor.shape[0]
    except AssertionError as err:
        raise Exception( f"to shuffle the training-data and -labels tensors,\nthose two tensors need to have the first number of instance\n(first dimension): {data_tensor.shape[0]} != {labels_tensor.shape[0]}" ) from err

    random.seed( seed )
    instance_label_pairs = list( zip( data_tensor, labels_tensor ) )
    # shuffles inplace, maintains features:label pairing 
    random.shuffle( instance_label_pairs ) 
    training_data_tup, training_labels_tup = zip( *instance_label_pairs )
    
    # restores tensors
    training_data, training_labels = torch.stack( training_data_tup ), torch.stack( training_labels_tup )
    del training_data_tup, training_labels_tup

    return training_data, training_labels

def main( cli_args, device, shuffle=True, verbose=True ):
    
    
    training_data = load_training_data( cli_args.training_dataset_path, as_tensor=True )    
    training_labels = load_training_labels( cli_args.training_labels_path, as_tensor=True )
    tensor_dataset = TensorDataset( training_data, training_labels )
    
    batch_size = cli_args.batch_size
    if batch_size == 0:
        batch_size = training_data.shape[0]

    tensor_dl = DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    if verbose:
        print( "data has been loaded")

    model = Mnist_CNN( ).to( device )
    
    # parametize this
    optimizer = optim.SGD( 
        model.parameters(), 
        lr=args.lr, 
        momentum=args.momentum 
    )

    # parametize this
    criterion = F.cross_entropy

    if verbose:
        print( "model, optimizer, and criterion have been defined" )

    losses = []
    if verbose:
        print( "starting the run!" )

    for epoch in range( cli_args.epochs ):

        model.train()
        for batchidx, ( data_instance, data_label ) in enumerate( tensor_dl ):
            # unsqueeze(0) adds a dimension in the leftmost position to deal with the Channels argument of the Conv2d layers
            unsqzd_di = data_instance.unsqueeze( 0 )
            preds = model( unsqzd_di ) 
            # max_class_vals, max_class_inds = torch.max( preds, 1 )
            loss = criterion( preds, data_label.long() ) # the .long typecast is a bandaid fix, idfk wtf is happening 
            losses.append( loss )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batchidx % cli_args.log_interval == 0:
                print( f"training epoch {epoch} / {cli_args.epochs}, batch #{batchidx} / {training_data.shape[0] // batch_size}\nLoss:\t{loss}\n" )

            
            
if __name__ == '__main__':
    # Training settings; I kept mostly the same names as those
    # in mnist_example_cnn.py for ease of comparison, but added arguments
    # and changed some default values.
    
    parser = argparse.ArgumentParser( description='Running Training Example' )
    
    parser.add_argument( '--training-dataset-path', type=str, default="train_images.pkl",
                        help="path to the training dataset pickle file (default: train_images.pkl)" )
    parser.add_argument( '--training-labels-path', type=str, default="train_labels.csv",
                        help="path to the training labels csv file (default: train_labels.csv)" )
    parser.add_argument( '--batch-size', type=int, default=0, metavar='N',
                        help='input batch size for training (default: 0, meaning entire dataset)' )
    parser.add_argument( '--test-batch-size', type=int, default=0, metavar='N',
                        help='input batch size for testing (default: 0, meaning entire dataset)' )
    parser.add_argument( '--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)' )
    parser.add_argument( '--lr', type=float, default=10E-03, metavar='LR',
                        help='learning rate (default: 10E-04)' )
    parser.add_argument( '--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)' )
    parser.add_argument( '--no-cuda', action='store_true', default=True,
                        help='disables CUDA training (default: True)' )
    parser.add_argument( '--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)' )
    parser.add_argument( '--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument( '--save-model', action='store_true', default=False,
                        help='For Saving the current Model (default: False)' )
    
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed( args.seed )

    device = torch.device( "cuda" if use_cuda else "cpu" )

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    main( args, device )