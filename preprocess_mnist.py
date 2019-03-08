import os
import csv
import pickle
import numpy as np
import scipy.misc
from sklearn.preprocessing import MinMaxScaler
from skimage.transform import resize
from tqdm import trange
DATA_FOLDER = 'mnist_photos'

pickle_in = open("train_images.pkl","rb")
image_array = pickle.load(pickle_in)

with open('train_labels.csv') as csv_file:
    labels = list(csv.reader(csv_file, delimiter=','))

header = labels.pop(0)

# make parent subdirectory for all of the data
os.makedirs(DATA_FOLDER, exist_ok=True)
# making the subdirectories for each label
label_categories = set()
for lb_pair in labels:
    # get indices converted from strings to ints
    lb_pair[0] = int(lb_pair[0])
    # add the label to the set
    lb = lb_pair[1]
    label_categories.add(lb)

for lb in label_categories:
    os.makedirs(os.path.join(DATA_FOLDER, str(lb)), exist_ok=True)

# save images to appropriate subfolder, with preprocessing for shape and channels
print('Preprocessing images')
label_counts = [0] * len(label_categories)
max_samples = 1000
for i in trange(len(labels)):
    lb = labels[i]
    if label_counts[int(lb[1])] < max_samples:
        label_counts[int(lb[1])] += 1
        image = image_array[lb[0]]
        # normalize image to (-1, 1)
        NewMin = -1.0
        NewMax = 1.0
        OldMin = 0
        OldMax = 255

        OldRange = (OldMax - OldMin)
        NewRange = (NewMax - NewMin)
        image = (((image - OldMin) * NewRange) / OldRange) + NewMin  # scale image to between -1 and 1

        # resize image to inception input shape
        image = resize(image, [256, 256])

        # change that shit back
        NewMin = 0
        NewMax = 255
        OldMin = -1.0
        OldMax = 1.0
        OldRange = (OldMax - OldMin)
        NewRange = (NewMax - NewMin)
        image = np.array((((image - OldMin) * NewRange) / OldRange) + NewMin, dtype=np.float32)  # scale image to between 0 and 255

        # triplicate image to 3 channels
        image = np.dstack((image, image, image))
        image_name = 'image_' + str(lb[0]) + '.png'
        image_dir = os.path.join(DATA_FOLDER, lb[1])
        scipy.misc.toimage(image, cmin=0, cmax=255).save(os.path.join(image_dir, image_name))
        # np.array(Image.open('mnist/9/image_0.jpg'))  # test that all values are the same


print('done')
