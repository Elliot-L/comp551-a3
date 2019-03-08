import os, pickle, torch
import pandas as pds 
import numpy as np

def load_training_labels( filepath=os.path.join( os.getcwd(), 'train_labels.csv' ), as_tensor=False ):
    """
    Wrapper around pds.read_csv with the appropriate args.

    Arguments:

        filepath: path to train_labels.csv 

    Returns:

        pandas DataFrame representation of the file (dtype = int64 ).
    """
    labels = pds.read_csv( 
        filepath, 
        sep=',',
        header=0,
        index_col=None
    )

    if as_tensor:
        return torch.tensor( np.array( labels.Category.values, dtype=np.float32 ) )
    else:
        return labels
    

def load_training_data( filepath=os.path.join( os.getcwd(), 'train_images.pkl' ), as_tensor=False ):
    """
    Wrapper around pickle.load with the appropriate args.

    Arguments:

        filepath: path to train_labels.csv 

    Returns:

        numpy 3D array representation of the file (dtype = float64 ).
    """
    with open( filepath, 'rb' ) as handle:
        data = pickle.load( handle )
    
    if as_tensor:
        return torch.tensor( data )
    
    else:
        return data
