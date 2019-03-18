# comp551-a3

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
    program called from the command line to run a meta-classifier on the predictions of >= 1 CNN model(s)

# barebones_run_on_test.py
functions:
    run_on_test, compute_features_only
classes:
    none
description:
    program called from the command line to run a model on a validation or testing dataset and record it's classification output.

# modd_barebones_runner.py