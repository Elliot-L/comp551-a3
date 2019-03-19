# comp551-a3

# logger.py
functions: 
    none
classes:
    Logger, useful wrapper for TensorFlow's Tensorboard display
description:
    used to use TensorFlow-like tensorboards in visualization.

# biggest_bbox_extractor.py
functions:
    get_all_rotations, cut_out_dom_bbox
classes:
    none
description:
    used for preprocessing the input images.

# data_loader.py
functions:
    load_training_labels, load_training_data, load_testing_data
classes:
    none
description:
    used to load the data in a convenient format.

# some_model_classes.py
functions:
    none
classes:
    the class definitions of all the neural network architectures we implemented, tweaked, and trained from scratch
description:
    file with all the models we implemented, tweaked, and trained from scratch.

# knn_metaclassifier.py
functions:
    fit_classifier, run_on_testing_set, merge_outputs, run_on_validation
classes:
    none
description:
    program called from the command line to run a meta-classifier on the saved predictions of >= 1 CNN model(s)

# barebones_run_on_test.py
functions:
    run_on_test, compute_features_only
classes:
    none
description:
    program called from the command line to run a model on a validation or testing dataset and record it's classification output.

# modd_barebones_runner.py
functions:
    train, validate, sanity_check_train, sanity_check_validate
classes:
    none
description:
    program called from the command line to run the Elliot-CNN models on the training set with a validation set
    and record it's classification output (and perform sanity-check runs on pytorch's MNIST dataset).

# preprocess_utils.py
functions:
    preprocess_image
classes:
    none
description:
    utility function for preprocessing images in preparation for use with the Inception fine-tuning script


# preprocess_mnist.py
functions:
    main
classes:
    none
description:
    Runs the full preprocess functionality on the provided pickle file of training data, transforming each input image
    using functionality from preprocess_utils and storing the resulting images in the directory structure needed for the
    Inception retrain.py script

# retrain.py
# taken from tensorflow tutorial
functions:
    main + helper functions
classes:
    none
description:
    Run from command line to do a fine-tuning of Inception on a new dataset. Requires the --image_dir parameter to be
    pointing to the output folder used by preprocess_mnist. If bottleneck files have not been created yet, it first
    runs the images through Inceptions pre-trained weights to produce bottleneck feature vectors. It then uses the
    cached bottleneck files to train a fully connected layer producing logits equal in number to the target classes.
    Significant hyperparameters can be set via command line arguments. Periodically evaluates on a validation subset
    of the training data, after training for the set number of steps will save the best model as a .pb file

# sample workflow
1. get the training data, testing data, and training labels in the same directory as these python files.
2. create the "logs" and "pickled-params" subdirectories in the same directory as these python files.
3. run 
$ python modd_barebones_runner.py --save-model --save-loaders True
4. run (for as many CNN models as you'd wish) 
$ python modd_barebones_runner.py --save-model --training-loader-pickle <path to pickled training loader> --validating-loader-pickle <path to pickled validation loader>
5. run 
$ python barebones_run_on_test.py --path-to-model-savefile <space-delimited absolute paths to all the *model.savefile in ./pickled-params> 
6. run 
$ python knn_metaclassifier.py --training-array-pickle-path <space-delimited absolute paths to all the training-dataset output feature files created in step 3/4> --testing-array-pickle-path <space-delimited absolute paths to all the validation-dataset output feature files created in step 3/4/5>