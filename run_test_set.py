import os
import numpy as np
import tensorflow as tf

GRAPH_PATH = '/tmp/results_padding_64/output_graph.pb'
DATA_PATH = '/tmp/bottleneck_test/dummy_folder0'
TEST_RESULTS_DIR = 'test_output'
def load_graph(filename):
    # some code taken from https://github.com/wisdal/Image-classification-transfer-learning/blob/master/test.ipynb
    #Unpersists graph from file as default graph."""
    with tf.gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


def run_graph(src, dest, input_layer_name, output_layer_name):
    # some code taken from https://github.com/wisdal/Image-classification-transfer-learning/blob/master/test.ipynb
    with tf.Session() as sess:
        i=0
        #outfile=open('submit.txt','w')
        #outfile.write('image_id, label \n')
        for f in os.listdir(src):
            # image_data=load_image(os.path.join(src,test[i]+'.jpg'))
            # #image_data=load_image(os.path.join(src,f))
            softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
            bottleneck_string = open(os.path.join(src, f)).read()
            bottleneck_tensor = np.array(bottleneck_string.split(','), dtype=np.float32)
            predictions, = sess.run(softmax_tensor, {input_layer_name: np.reshape(bottleneck_tensor, (1, 2048))})
            #
            # # Sort to show labels in order of confidence
            # top_k = predictions.argsort()[-num_top_predictions:][::-1]
            # for node_id in top_k:
            #     human_string = labels[node_id]
            #     score = predictions[node_id]
            #     #print('%s (score = %.5f) %s , %s' % (test[i], human_string))
            #     print('%s, %s' % (test[i], human_string))
            #     #outfile.write(test[i]+', '+human_string+'\n')
            i+=1


load_graph(GRAPH_PATH)
# input_layer_name='DecodeJpeg/contents:0'  # for decoded inputs
# input_layer_name='input'  # for a straight up image, but doesn't work
input_layer_name = 'input/BottleneckInputPlaceholder:0'
run_graph(src=DATA_PATH, dest=TEST_RESULTS_DIR, input_layer_name=input_layer_name, output_layer_name='final_result:0')
print('done')