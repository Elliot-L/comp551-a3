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
from data_loader import load_training_data, load_training_labels, load_testing_data
from some_model_classes import *
from biggest_bbox_extractor import cut_out_dom_bbox

def run_on_test( model, device, test_loader, output_file_path, number_of_padding_arrays ):
    """
    Runs model on testing set, saves output in output_file_path
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for i, data in tqdm( enumerate( test_loader ) ): # iterates through batches of 64 64x64 images
            data = data.unsqueeze( 1 )
            data = data.to( device )
            output = model( data ) # make classification
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            for batch_element in range( test_loader.batch_size ):
                preds.append( f"{batch_element + ( i * test_loader.batch_size )},{pred[ batch_element ].item()}" )
    
    with open( output_file_path, 'w' ) as predictions_file:
        predictions_file.write( 'Id,Category\n' )
        predictions_file.write( '\n'.join( preds[:-number_of_padding_arrays] ) ) 
    
    print( f">>> Wrote predictions in {output_file_path} \n" )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs a PyTorch model on the test Modified MNIST dataset')
    parser.add_argument('--path-to-model-savefile', type=str, required=True,
                        help="path to the model save file, should be in /pickled-params/<something>/<timestamp>_model.savefile")
    parser.add_argument('--model-batch-size', type=int, default=64,
                        help='batch size for model (default: 64)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # hard-coded parameters start here
    model = Other_MNIST_CNN().to( device ).double() # casting it to double because of some  weird pytorch peculiarities
    
    # useful reference for debugging
    # batch_size = 64
    # path_to_mode_savefile = os.path.abspath( r"C:\Users\Samy\Dropbox\Samy_Dropbox\MSc\winter-2019-courses\COMP-551\mini-project-3\comp551-a3\pickled-params\2019-03-13_22-33_model.savefile" )
    # hard-coded parameters end here
        
    # load learned parameters
    
    model.load_state_dict( torch.load( args.path_to_model_savefile ) )
    #model.load_state_dict( torch.load( path_to_mode_savefile ) )
    print( ">>> Loaded model\n\n" )

    test_dataset, number_of_padding_arrays = load_testing_data( as_tensor=True )
    print( ">>> Loaded test dataset" )

    preprocessed_images = [ cut_out_dom_bbox( test_dataset[i,:,:], as_tensor=True )[0] for i in range( test_dataset.shape[0] ) ] # avoids trying to find bounding box in padding all-0 arrays
    print( ">>> Preprocessed test dataset images" )

    test_dataset = torch.stack( preprocessed_images )
    print( f">>> {test_dataset.shape}" )

    test_loader = torch.utils.data.DataLoader( 
        test_dataset,
        batch_size=args.model_batch_size,
        #batch_size=batch_size,
        shuffle=False
    )
    
    output_file_path = os.path.join( os.path.dirname( path_to_mode_savefile ), 'test_set_predictions.csv' ) 

    print( ">>> Evaluating on test dataset" )
    run_on_test( model, device, test_loader, output_file_path, number_of_padding_arrays )
   
    print( ">>> Finished" )