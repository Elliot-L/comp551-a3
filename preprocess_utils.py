import numpy as np
from biggest_bbox_extractor import cut_out_dom_bbox


def preprocess_image(sample, zero_padding=True, bbox_size=64, change_range=False):
    # trying the resized bounding box of the dominant number
    image = cut_out_dom_bbox(sample, output_dim=bbox_size)[0]  # scales image between 0 and 1

    # change range back to (0, 255)
    if change_range:
        NewMin = 0
        NewMax = 255
        OldMin = 0.0
        OldMax = 1.0
        OldRange = (OldMax - OldMin)
        NewRange = (NewMax - NewMin)
        image = np.array((((image - OldMin) * NewRange) / OldRange) + NewMin,
                         dtype=np.float32)  # scale image to between 0 and 255

    inception_dimension = 299  # magic number, inception takes (299, 299) dim input tensors
    # going to experiment with zero padding images to the full resolution so that the digit doesn't get further upscaled
    if zero_padding:
        top_pad = int((299 - image.shape[0]) / 2)
        bottom_pad = int((299 - image.shape[0]) / 2) + 1  # we rounded down by doing the int division, 299 is odd
        left_pad = int((299 - image.shape[1]) / 2)
        right_pad = int((299 - image.shape[1]) / 2) + 1  # same situation as bottom_pad
        image = np.pad(image, [(top_pad, bottom_pad), (left_pad, right_pad)], mode='constant')

    # triplicate image to 3 channels
    image = np.dstack((image, image, image))
    return image