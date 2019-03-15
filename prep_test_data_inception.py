import os
import pickle
import numpy as np
import scipy.misc
from tqdm import trange
from preprocess_utils import preprocess_image

DATA_FOLDER = 'test_mnist_photos'

pickle_in = open("train_images.pkl","rb")
image_array = pickle.load(pickle_in)

# make parent subdirectory for all of the data
os.makedirs(DATA_FOLDER, exist_ok=True)

for i in trange(len(image_array)):
    image = preprocess_image(image_array[i], zero_padding=True, bbox_size=64)
    image_name = 'image_' + str(i) + '.png'
    scipy.misc.toimage(image, cmin=0, cmax=255).save(os.path.join(DATA_FOLDER, image_name))