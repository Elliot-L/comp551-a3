import importlib, importlib.util, os.path, random
import numpy as np

def load_data(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def split_data(s):

    random.seed(s)
    train_idx = random.sample(range(40000), 28000)
    test_idx = set(range(0, 40000)) - set(train_idx)

    return train_idx, test_idx

# load data and labels from data_loader
data_loader = load_data("data_loader", "data_loader.py")
data = data_loader.load_training_data()
labels = data_loader.load_training_labels()

# set seed
s = 2000
# get index for train and test
train_idx, test_idx = split_data(s)

# get the training and test data and labels
train_data = data[train_idx]
train_labels = np.asarray(labels)[train_idx]
test_data = data[list(test_idx)]
test_labels = np.asarray(labels)[list(test_idx)]
