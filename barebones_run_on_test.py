# warning: VSC is weird with torch
# README: modified version of barebones_runner.py to run on our dataset.

from __future__ import print_function

import os, argparse, pickle

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

# from tensorboardX import SummaryWriter

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

def compute_features_only( model, device, test_loader, number_of_padding_arrays ):

    model.eval()
    outputs, instance_row_indx_list = None, []
    with torch.no_grad():
        for batch_idx, data in tqdm( enumerate( test_loader ) ):
            data = data.unsqueeze( 1 )
            data = data.to( device )
            output = model( data ) # make classification

            if batch_idx == 0:
                outputs = output.data.numpy()
                instance_row_indx_list.extend( [ batch_element_indx + ( batch_idx * test_loader.batch_size ) for batch_element_indx in range( test_loader.batch_size ) ] )
            else:
                outputs = np.vstack( ( outputs, output.data.numpy() ) )
                instance_row_indx_list.extend( [ batch_element_indx + ( batch_idx * test_loader.batch_size ) for batch_element_indx in range( test_loader.batch_size ) ] )

    return outputs[:-number_of_padding_arrays,:], instance_row_indx_list[:-number_of_padding_arrays]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs a PyTorch model on the test Modified MNIST dataset')
    parser.add_argument('--path-to-model-savefile', nargs='+', type=str, required=True,
                        help="path to the model save file, should be in /pickled-params/<something>/<timestamp>_model.savefile")
    parser.add_argument('--model-batch-size', type=int, default=64,
                        help='batch size for model (default: 64)')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print( ">>> Have you checked that the model you are using is the same model as the one(s) you trained with?" )
    # useful reference for debugging
    # batch_size = 64
    # path_to_model_savefile = os.path.abspath( r"C:\Users\Samy\Dropbox\Samy_Dropbox\MSc\winter-2019-courses\COMP-551\mini-project-3\comp551-a3\pickled-params\2019-03-13_22-33_model.savefile" )
    # hard-coded parameters end here
    
    test_dataset, number_of_padding_arrays = load_testing_data( as_tensor=True )
    print( ">>> Loaded test dataset" )

    preprocessed_images = [ cut_out_dom_bbox( test_dataset[i,:,:], as_tensor=True )[0] for i in range( test_dataset.shape[0] ) ] # avoids trying to find bounding box in padding all-0 arrays
    print( ">>> Preprocessed test dataset images" )

    test_dataset = torch.stack( preprocessed_images )
    print( f">>> {test_dataset.shape}" )

    test_loader = torch.utils.data.DataLoader( 
        test_dataset,
        batch_size=args.model_batch_size,
        shuffle=False
    )

    start_timestamp = datetime.now().strftime( '%Y-%m-%d_%H-%M' )

    # load learned parameters
    if len( args.path_to_model_savefile ) == 1:
        # hard-coded parameters start here
        model = Elliot_Model().to( device ).double() # casting it to double because of some  weird pytorch peculiarities
        # hard-coded parameters end here
        model.load_state_dict( torch.load( args.path_to_model_savefile ) )

        print( ">>> Loaded model\n\n" )
        
        output_file_path = os.path.join( os.path.dirname( args.path_to_model_savefile ), f'{start_timestamp}_test_set_predictions.csv' ) 

        print( ">>> Evaluating on test dataset" )
        _ = run_on_test( model, device, test_loader, output_file_path, number_of_padding_arrays )
    
        print( ">>> Finished" ) 
    
    else:
        
        output_file_path = os.path.join( os.path.dirname( args.path_to_model_savefile[0] ), f'{start_timestamp}_test_set_features_array.pickle' ) 

        compiled_features_array = None
        
        for e, model_savefile in enumerate( args.path_to_model_savefile ):
            # hard-coded arguments start here
            model = Elliot_Model().to( device ).double() # casting it to double because of some  weird pytorch peculiarities
            model.load_state_dict( torch.load( model_savefile ) )
            
            print( f">>> Loaded model {e+1} / {len( args.path_to_model_savefile)}\n" )
            
            if e == 0:
                model_outputs, instance_row_indx_list = compute_features_only( model, device, test_loader, number_of_padding_arrays )
                input( model_outputs.shape )
                input( len( instance_row_indx_list ) )
                compiled_features_array = np.hstack( ( model_outputs, np.array( instance_row_indx_list ).reshape( len( instance_row_indx_list ), -1 ) ) )
            
            else:
                model_outputs, _ = compute_features_only( model, device, test_loader, number_of_padding_arrays )
                compiled_features_array = np.hstack( ( model_outputs, compiled_features_array ) )
        
        
        with open( output_file_path, 'wb' ) as predictions_file:
            pickle.dump( compiled_features_array, predictions_file, protocol=pickle.HIGHEST_PROTOCOL )
        
        print( f">>> Saved feature array (with the row indx as the last row) in:\n{output_file_path}")


