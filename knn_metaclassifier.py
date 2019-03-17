import pickle, os, inspect, argparse

import numpy as np

from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier

from data_loader import load_testing_data
from datetime import datetime

def fit_classifier( array ):
    
    features = array[:,:-1]
    labels = array[:,-1]
    knn_clf = KNeighborsClassifier( n_neighbors=5, weights='distance' )
    knn_clf.fit( features, labels )
    return knn_clf 

def run_on_validation_this_needs_to_change( clf, array, output_filepath=None ):
    
    features = array[:,:-1]
    targets = array[:,-1]
    classifications = clf.predict( features )

    if output_filepath is None:

        start_timestamp = datetime.now().strftime( '%Y-%m-%d_%H-%M' )
        output_filepath = os.path.join( os.path.dirname( output_filepath ), f'{start_timestamp}_knn_predictions.csv' )
    
    # getting the arguments' values
    current_frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues( current_frame )
    args_and_vals = ','.join( [ ( arg, values[arg] ).__repr__() for arg in args ] )

    pred_class_target_pair = [ ( cl, tar ) for ( cl, tar ) in zip( classifications, targets ) ]

    with open( output_filepath, 'w' ) as outfile:
        outfile.write( f"#{args_and_vals}\n" )
        for ( id,cl ) in pred_class_target_pair:
            outfile.write( f"{id},{cl}\n" )
    
    print( f">>> Saved KNN classifications in:\n{output_filepath}" )

    correct = sum( [ 1 for ( cl, tar ) in pred_class_target_pair if cl == tar ] )

    print( f">>> Accuracy: {correct} / {len( pred_class_target_pair )} = { correct / len( pred_class_target_pair )}")


def run_on_testing_set( clf, array, output_filepath=None ):
    
    features = array[:,:-1]
    classifications = clf.predict( features )

    if output_filepath is None:

        start_timestamp = datetime.now().strftime( '%Y-%m-%d_%H-%M' )
        output_filepath = os.path.join( os.path.dirname( output_filepath ), f'{start_timestamp}_knn_predictions.csv' )
    
    # getting the arguments' values
    current_frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues( current_frame )
    args_and_vals = ','.join( [ ( arg, values[arg] ).__repr__() for arg in args ] )

    pred_class_target_pair = [ ( i, int( cl ) ) for i, cl in enumerate( classifications ) ]

    with open( output_filepath, 'w' ) as outfile:
        outfile.write( f"#{args_and_vals}\n" )
        outfile.write("Id,Category\n")
        for ( instance, cl ) in pred_class_target_pair:
            outfile.write( f"{instance},{cl}\n" )
    
    print( f">>> Saved KNN classifications in:\n{output_filepath}" )

def merge_outputs( output_pickled_array_filepaths ):

    merged_arr = None
    for e, filepath in enumerate( output_pickled_array_filepaths ):
        with open( filepath, 'rb' ) as handle:
            this_models_output = pickle.load( handle )
        if e == 0:
            merged_arr = this_models_output[:]
        else:
            merged_arr = np.hstack( ( this_models_output[:,:-1], merged_arr ) ) # the -1 is to skip the last column (the actual class target column)
    return merged_arr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KNN meta-classifier')
    parser.add_argument('--training-array-pickle-path', nargs='+', type=str, metavar='F', required=True,
                        help='path to the pickled # training instances X ( # models * 10 ) + 1 np.ndarray.')
    parser.add_argument('--testing-array-pickle-path', type=str, metavar='T', required=True,
                        help='path to the pickled # testing instances X ( # models * 10 ) + 1 np.ndarray.')
    parser.add_argument('--output-filepath', type=str, metavar='O', 
                        help='destination path for classification file.')    
    args = parser.parse_args()

    training_array = merge_outputs( args.training_array_pickle_path )
    print( f">>> The training array's shape is {training_array.shape}\n")
    # if running with validation set
    # validation_array = merge_outputs( args.testing_array_pickle_path )
    with open( args.testing_array_pickle_path, 'rb' ) as handle:
        testing_array = pickle.load( handle )
    
    knn_clf = fit_classifier( training_array )

    print( ">>> Finished training KNN classifier" )

    run_on_testing_set( knn_clf, testing_array, output_filepath=os.path.join( os.getcwd(), 'first_knn_metaclassification.csv' ) )

