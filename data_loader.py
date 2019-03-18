import os, pickle, torch, random
import pandas as pds 
import numpy as np

from tqdm import tqdm

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
        return torch.tensor( np.array( labels.Category.values, dtype=np.float32 ), dtype=torch.long )
    else:
        return labels.Category.values


def load_training_data(filepath=os.path.join(os.getcwd(), 'train_images.pkl'), as_tensor=False):
    """
    Wrapper around pickle.load with the appropriate args.

    Arguments:

        filepath: path to train_labels.csv
        return_rotated: will cause the returned data to have 4 channels instead of 1, channels 2,3,4 with rotated images

    Returns:

        numpy 3D array/pytorch tensor representation of the file (dtype = float64 ).
    """
    with open( filepath, 'rb' ) as handle:
        data = pickle.load( handle )

    if as_tensor:
        return torch.tensor( data )
    
    else:
        return data

def load_testing_data( filepath=os.path.join( os.getcwd(), 'test_images.pkl' ), as_tensor=False, pad_to_multiple_of=64 ):
    """
    Wrapper around pickle.load with the appropriate args and a padding option.

    Arguments:

        filepath: path to train_labels.csv 

        as_tensor: boolean indicator of whether to return a pytorch tensor (True) or numpy array (False).

        pad_to_multiple_of: int, enforced the returned dataset's first dimension to be a multiple of this parameter 
                            (adds all-0 arrays until that is the case)

    Returns:

        numpy 3D array/pytorch tensor representation of the file (dtype = float64 )

        number of all-0 arrays that were appended to the tail of the 3D numpy array/pytorch tensor.
    """
    with open( filepath, 'rb' ) as handle:
        data = pickle.load( handle )

    number_of_zero_64by64_arrays_to_add = pad_to_multiple_of - ( data.shape[0] % pad_to_multiple_of )
    pad_arrays = np.zeros( ( number_of_zero_64by64_arrays_to_add, 64, 64), dtype=float )
    dataset = np.concatenate( ( data, pad_arrays ), axis=0 )
    
    print( f"Note: dataset had {data.shape[0]} images, added {number_of_zero_64by64_arrays_to_add} zero arrays to get a dataset of shape {dataset.shape}" )

    if as_tensor: 

        return torch.tensor( dataset ), number_of_zero_64by64_arrays_to_add
    
    else:
        return dataset, number_of_zero_64by64_arrays_to_add