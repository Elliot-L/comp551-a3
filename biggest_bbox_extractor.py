import pickle, torch
import matplotlib.pyplot as plt
import numpy as np 

from operator import itemgetter
from skimage.measure import find_contours
# from skimage.util import invert
from skimage.transform import resize


def cut_out_dom_bbox( arr:np.ndarray, digit_color_threshold=0.9, output_dim=64, bbox_offset=2, as_tensor=True ):
    """
    Finds contours in input image and returns the largest bounding box in the image.

    Arguments:

        arr: numpy array of values in range [ 0, 255 ].

        digit_color_threshold: "level" arg to pass onto skimage.measure.find_contours.

        output_dim: the dimension of the output array (uses skimage.transform.resize).

        bbox_offset: number representing the width of the margin surrounding the bounding box in the output array (in pixels).

    Returns:

        img_arr[ xmin:xmax, ymin:ymax ]: subarray of input arr containing the largest bounding box, resized to shape=( output_dim x output_dim ).

        ( xmin, xmax ), ( ymin, ymax ): two tuples representing the corners of the largest bounding box.
    """

    # rescales image from [ 0, 255 ] -> [ 0.0, 1.0 ]
    img_arr = np.copy( arr ) / 255

    if np.sum( img_arr ) == 0 :
        if as_tensor:
            return torch.tensor( img_arr ), None, None
        else:
            return img_arr, None, None
    
    
    # contour_vertices is a list of [ y, x ] arrays, where y and x are floats
    contour_vertices = find_contours( img_arr, digit_color_threshold )

    # list of ( contour_ind, ( xmin, xmax ), ( ymin, ymax ), dimension ) tuples
    bboxes = []
    for sil_i, silhouette in enumerate( contour_vertices ):
        bbox_xmax, bbox_ymax = np.rint( np.amax( silhouette, axis=0 ) )
        bbox_xmax += 1 # for inclusion of the max x-value
        bbox_ymax += 1 # for inclusion of the max y-value
        bbox_xmin, bbox_ymin = np.rint( np.amin( silhouette, axis=0  ) )

        bbox_dim = max( ( bbox_xmax - bbox_xmin ), ( bbox_ymax - bbox_ymin ) ) 
        bboxes.append( ( sil_i, ( int( bbox_xmin ), int( bbox_xmin+bbox_dim ) ), ( int( bbox_ymin ), int( bbox_ymin+bbox_dim ) ), bbox_dim ) )
    
    bboxes.sort( key=lambda x:x[-1], reverse=True ) # in-place sort from highest bbox_dim -> lowest bbox_dim 

    ( xmin, xmax ), ( ymin, ymax ) = bboxes[0][1], bboxes[0][2]
    xmin -= int( bbox_offset/2 )
    ymin -= int( bbox_offset/2 )
    xmin, ymin = max( 0, xmin ), max( 0, ymin )
    xmax += int( bbox_offset/2 )
    ymax += int( bbox_offset/2 )

    xmax, ymax = min( output_dim, xmax ), min( output_dim, ymax )
    if as_tensor:
        return torch.tensor( resize( img_arr[ xmin:xmax, ymin:ymax ], ( output_dim, output_dim ) ) ), ( xmin, xmax ), ( ymin, ymax ) 
    else:
        return resize( img_arr[ xmin:xmax, ymin:ymax ], ( output_dim, output_dim ) ), ( xmin, xmax ), ( ymin, ymax )

if __name__ == '__main__':
    with open("train_images.pkl", "rb") as handle:
        data = pickle.load( handle )

    test_img = data[0]
    dom_dig, xpoints, ypoints = cut_out_dom_bbox( test_img )

    fig = plt.figure()
    ax = fig.subplots( nrows=1, ncols=2 )
    ax[0].imshow( test_img, cmap='binary' )
    ax[0].scatter( ypoints[0], xpoints[0], color='red' )
    ax[0].scatter( ypoints[0], xpoints[1], color='red' )
    ax[0].scatter( ypoints[1], xpoints[0], color='red' )
    ax[0].scatter( ypoints[1], xpoints[1], color='red' )
    ax[1].imshow( dom_dig, cmap='binary' )
    
    plt.show()