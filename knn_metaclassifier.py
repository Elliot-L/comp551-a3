import pickle, os, inspect, argparse

from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier

from data_loader import load_testing_data
from datetime import datetime

def fit_classifier( path_to_pickled_array ):
    with open( path_to_pickled_array, 'rb' ) as handle:
        p = pickle.load( handle )
    features = p[:,:-1]
    labels = p[:,-1]
    knn_clf = KNeighborsClassifier( n_neighbors=5, weights='distance' )
    knn_clf.fit( features, labels )
    return knn_clf 

def run_on_testing_set( clf, path_to_pickled_array, output_filepath=None ):
    with open( path_to_pickled_array, 'rb' ) as handle:
        p = pickle.load( handle )
    features = p[:,:-1]
    instance_row_indx = p[:,-1]
    classifications = clf.predict( features )

    if output_filepath is None:

        start_timestamp = datetime.now().strftime( '%Y-%m-%d_%H-%M' )
        output_filepath = os.path.join( os.path.dirname( path_to_pickled_array ), f'{start_timestamp}_knn_predictions.csv' )
    
    # getting the arguments' values
    current_frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues( current_frame )
    args_and_vals = ','.join( [ ( arg, values[arg] ).__repr__() for arg in args ] )

    instance_indx_classification_pair = [ ( iri, cl ) for ( iri, cl ) in zip( instance_row_indx, classifications ) ]
    instance_indx_classification_pair.sort( key=lambda x:x[0] )

    with open( output_filepath, 'w' ) as outfile:
        outfile.write( f"#{args_and_vals}\n" )
        for ( id,cl ) in instance_indx_classification_pair:
            outfile.write( f"{id},{cl}\n" )
    
    print( f">>> Saved KNN classifications in:\n{output_filepath}" )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KNN meta-classifier')
    parser.add_argument('--training-array-pickle-path', type=str, metavar='F', required=True,
                        help='path to the pickled # training instances X ( # models * 10 ) + 1 np.ndarray.')
    parser.add_argument('--testing-array-pickle-path', type=str, metavar='T', required=True,
                        help='path to the pickled # testing instances X ( # models * 10 ) + 1 np.ndarray.')
    parser.add_argument('--output-filepath', type=str, metavar='O', 
                        help='destination path for classification file.')    
    args = parser.parse_args()

    knn_clf = fit_classifier( args.training_array_pickle_path )

    print( ">>> Finished training KNN classifier" )

    run_on_testing_set( knn_clf, args.testing_array_pickle_path, output_filepath=None )
