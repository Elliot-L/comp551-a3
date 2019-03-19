import pickle, os, inspect, argparse

import numpy as np

from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

from data_loader import load_testing_data
from datetime import datetime

def fit_classifier( array ):
    """
    Wrapper around scikit-learn's model.fit() function.

    Arguments:
        
        array: numpy ndarray whose columns are features (except the last column, which is the label).

    Returns:
        
        a trained KNN classifier.
    """
    features = array[:,:-1]
    labels = array[:,-1]
    knn_clf = KNeighborsClassifier( n_neighbors=11, weights='distance' )
    knn_clf.fit( features, labels )
    return knn_clf 

def run_on_validation_set( clf, array, output_filepath=None ):
    """
    Wrapper around scikit-learn's model.predict() function which also saves the output and other bookkeeping details.

    Arguments:

        clf: trained scikit learn classifier (KNN classifier).

        array: numpy ndarray whose columns are features (except the last column, which is the class label).

        output_filepath: path to the output file.
    
    Returns:

        nothing
    """
    features = array[:,:-1]
    targets = array[:,-1]
    classifications = clf.predict( features )

    if output_filepath is None:

        start_timestamp = datetime.now().strftime( '%Y-%m-%d_%H-%M' )
        output_filepath = os.path.join( os.getcwd(), f'{start_timestamp}_knn_predictions_on_validation.csv' )
    
    print( f">>> output file path is {output_filepath}" )
    # getting the arguments' values
    current_frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues( current_frame )
    args_and_vals = ','.join( [ ( arg, values[arg] ).__repr__() for arg in args ] )

    pred_class_target_pair = [ ( cl, tar ) for ( cl, tar ) in zip( classifications, targets ) ]
    
    print( confusion_matrix( targets, classifications ) )

    with open( output_filepath, 'w' ) as outfile:
        outfile.write( f"#{args_and_vals}\n" )
        for ( id,cl ) in pred_class_target_pair:
            outfile.write( f"{id},{cl}\n" )
    
    print( f">>> Saved KNN classifications in:\n{output_filepath}" )

    correct = sum( [ 1 for ( cl, tar ) in pred_class_target_pair if cl == tar ] )

    print( f">>> Accuracy: {correct} / {len( pred_class_target_pair )} = { correct / len( pred_class_target_pair )}")

def run_on_testing_set( clf, array, output_filepath=None ):
    """
    Wrapper around scikit-learn's model.predict() function which also saves the output and other bookkeeping details.

    Arguments:

        clf: trained scikit learn classifier (KNN classifier).

        array: numpy ndarray whose columns are features (except the last column, which is the row index of this row in the test set - used in case of unexpected shuffling).

        output_filepath: path to the output file.
    
    Returns:

        nothing
    """
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

def merge_outputs( output_pickled_array_filepaths, join_train_and_val=False ):
    """
    Merges the [ predictions, [label] ] arrays produced by individual CNN models into a single array.

    Arguments:

        output_pickled_array_filepaths: list of paths to the pickled [ predictions, [label] ] arrays.
    
    Returns:

        merged_arr: np.ndarray whose rows are instances, columns are each model in output_pickled_array_filepaths's predictions, 
                    and whose last column is the actual label of the instance.

    """
    if join_train_and_val:
        merged_arr = None
        train_and_val_output_pairs = list(
            zip( 
                output_pickled_array_filepaths[ : len( output_pickled_array_filepaths )//2 ], 
                output_pickled_array_filepaths[ len( output_pickled_array_filepaths )//2 : ] 
            ) 
        )
        print( train_and_val_output_pairs )
        
        for e, ( train_path, val_path ) in enumerate( train_and_val_output_pairs ):
            with open( train_path, 'rb' ) as handle:
                model_train_output = pickle.load( handle )
            with open( val_path, 'rb' ) as handle:
                model_val_output = pickle.load( handle )
            assert model_train_output.shape[1] == model_val_output.shape[1]

            if e == 0:
                merged_arr = np.vstack( ( model_train_output, model_val_output ) )
            else:
                this_models_merged_arr = np.vstack( ( model_train_output, model_val_output ) )[:,:-1]
                print( f"this model's shape = {this_models_merged_arr.shape}, merged_arr shape = {merged_arr.shape} " )
                
                merged_arr = np.hstack( ( this_models_merged_arr, merged_arr ) )
            print( f"merged_arr now has shape {merged_arr.shape}" )
        
        return merged_arr 
    else:
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
    parser.add_argument('--testing-array-pickle-path', nargs='+', type=str, metavar='T', required=True,
                        help='path to the pickled # testing instances X ( # models * 10 ) + 1 np.ndarray.')
    parser.add_argument('--output-filepath', type=str, metavar='O', 
                        help='destination path for classification file.')  
    parser.add_argument('--is-testset', type=bool, default=False,
                        help='whether the input testting-array-pickle-path represents the testing set (True) or the validation set (False)')
    args = parser.parse_args()

    training_array = merge_outputs( args.training_array_pickle_path, join_train_and_val=args.is_testset )
    print( f">>> The training array's shape is {training_array.shape}\n")
    # if running with validation set
    # validation_array = merge_outputs( args.testing_array_pickle_path )
    
    knn_clf = fit_classifier( training_array )

    if args.is_testset:
        with open( args.testing_array_pickle_path[0], 'rb' ) as handle:
            testing_array = pickle.load( handle )
    
        print( ">>> Finished training KNN classifier" )

        run_on_testing_set( knn_clf, testing_array, output_filepath=os.path.join( os.getcwd(), 'first_knn_metaclassification.csv' ) )

    else:
        validation_array = merge_outputs( args.testing_array_pickle_path )
        print( f">>> The validation array's shape is {validation_array.shape}\n" )
        run_on_validation_set( knn_clf, validation_array )