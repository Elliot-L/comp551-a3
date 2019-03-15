import os
import csv
import pickle
import numpy as np
import scipy.misc
from sklearn.preprocessing import MinMaxScaler
from skimage.transform import resize
from tqdm import trange
from preprocess_utils import preprocess_image
DATA_FOLDER = 'mnist_photos'

def main():
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
    max_samples = 5000
    for i in trange(len(labels)):
        lb = labels[i]
        if label_counts[int(lb[1])] < max_samples:
            label_counts[int(lb[1])] += 1
            image = image_array[lb[0]]
            image = preprocess_image(image, zero_padding=False, bbox_size=64, change_range=False)

            # # trying the resized bounding box of the dominant number
            # image = cut_out_dom_bbox(image, output_dim=128)[0]  # scales image between 0 and 1
            #
            # # change range back to (0, 255)
            # NewMin = 0
            # NewMax = 255
            # OldMin = 0.0
            # OldMax = 1.0
            # OldRange = (OldMax - OldMin)
            # NewRange = (NewMax - NewMin)
            # image = np.array((((image - OldMin) * NewRange) / OldRange) + NewMin, dtype=np.float32)  # scale image to between 0 and 255
            #
            # inception_dimension = 299 # magic number, inception takes (299, 299) dim input tensors
            # # going to experiment with zero padding images to the full resolution so that the digit doesn't get further upscaled
            # if ZERO_PADDING:
            #     top_pad = int((299 - image.shape[0]) / 2)
            #     bottom_pad = int((299 - image.shape[0]) / 2) + 1  # we rounded down by doing the int division, 299 is odd
            #     left_pad = int((299 - image.shape[1]) / 2)
            #     right_pad = int((299 - image.shape[1]) / 2) + 1  # same situation as bottom_pad
            #     image = np.pad(image, [(top_pad, bottom_pad), (left_pad, right_pad)], mode='constant')
            #
            # # triplicate image to 3 channels
            image = np.dstack((image, image, image))
            image_name = 'image_' + str(lb[0]) + '.png'
            image_dir = os.path.join(DATA_FOLDER, lb[1])
            scipy.misc.toimage(image, cmin=0, cmax=1).save(os.path.join(image_dir, image_name))
            # np.array(Image.open('mnist/9/image_0.jpg'))  # test that all values are the same

# run the preprocessor
main()