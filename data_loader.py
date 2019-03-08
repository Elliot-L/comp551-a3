import os, pickle
import pandas as pds 

def load_training_labels( filepath ):
    """
    Wrapper around pds.read_csv with the appropriate args.

    Arguments:

        filepath: path to train_labels.csv 

    Returns:

        pandas DataFrame representation of the file (dtype = int64 ).
    """
    return pds.read_csv( 
        filepath, 
        sep=',',
        header=0,
        index_col=None
    )

def load_training_data( filepath ):
    """
    Wrapper around pickle.load with the appropriate args.

    Arguments:

        filepath: path to train_labels.csv 

    Returns:

        numpy 3D array representation of the file (dtype = float64 ).
    """
    with open( filepath, 'rb' ) as handle:
        return pickle.load( handle )
