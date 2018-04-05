from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time
import numpy as np
import tensorflow as tf
from Unet3D import Unet
import os
from tensorflow.python.tools import freeze_graph

class TF(object):
    def __init__(self, gpu=False, shape=(1,3,128,128,128),
                 merge=False, symmetric=True, residual=True,
                 threads=44, optimize=False, activation="relu", batchnorm=True):
        """docstring for Tensorflow."""
        super(TF, self).__init__()

        device = "/GPU:0" if gpu else "/cpu:0"
        config = tf.ConfigProto()
        self.data_format='channels_first'
        #shape = [shape[0], shape[2], shape[3], shape[4], shape[1]]
        self.activation = tf.nn.relu if activation=="relu" else tf.nn.elu

        if not gpu:
            self.data_format='channels_last'
            shape = [shape[0], shape[2], shape[3], shape[4], shape[1]]
            config.intra_op_parallelism_threads = threads
            config.inter_op_parallelism_threads = threads

        # Creates a graph.

        with tf.device(device):
            images = tf.constant(np.random.rand(*shape), dtype=tf.float32)
            #self.outputs = self.simple_conv(images)
            self.outputs = Unet(merge=merge,
                                batchnorm=batchnorm,
                                data_format=self.data_format,
                                activation=activation,
                                symmetric=symmetric,
                                residual=residual).forward(images)
            self.outputs = tf.identity(self.outputs, name="output")

        if optimize:
            freeze(images, self.outputs)
            tf.reset_default_graph()

        # Creates a session with log_device_placement set to True.
        self.sess = tf.Session(config=config)
        if optimize:
            load_frozen_graph()
            saver = tf.train.import_meta_graph("dest/model.ckpt.meta", import_scope=None)
            saver.restore(self.sess, "dest/model.ckpt")
            self.outputs = "output:0"
        self.sess.run(tf.global_variables_initializer())
        #self.outputs = get_node_by_name(tf_graph, "output")

    def simple_conv(self, inputs, n=640, filters=1, kernel_size=1, strides=1, batchnorm=True):
        for i in range(n):
            inputs = tf.layers.conv3d(
                  inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                  padding=('SAME' if strides == 1 else 'VALID'), use_bias=True,
                  kernel_initializer=tf.variance_scaling_initializer(),
                  data_format=self.data_format)

            if batchnorm:
                inputs = tf.layers.batch_normalization(
                    inputs, fused=False)
            inputs =  self.activation(inputs)
        return inputs

    def process(self):
        t1 = time.time()
        self.sess.run(self.outputs)
        t2 = time.time()
        return t2-t1


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


def freeze(input_node, output_node, temp_folder=None):
    saver = tf.train.Saver()
    global_init = tf.global_variables_initializer()
    if temp_folder==None:
        temp_folder = os.path.dirname(os.path.realpath(__file__))+'/dest/'

    output_name = (output_node.name).split(":")[0]
    input_shape = input_node.get_shape().as_list()
    if input_shape[0] == -1:
        input_shape[0] = 1

    with tf.Session() as sess:
        sess.run(global_init)
        tf.train.write_graph(sess.graph, os.path.dirname(os.path.realpath(__file__)), 'dest/deploy.pbtxt', as_text=True)
        input = np.random.rand(*input_shape).astype(dtype=np.float32)
        #in_path = temp_folder+'/input'
        #input.tofile(in_path)

        out = sess.run(output_node, feed_dict={input_node:input})
        #out_path = temp_folder+'/output'
        #out.tofile(out_path)

        save_path = saver.save(sess,  temp_folder+'model.ckpt')
        #print('Model saved in file: {}'.format(save_path))
        # Look here for more details https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph_test.py
        freeze_graph.freeze_graph(
           os.path.join(temp_folder, 'deploy.pbtxt'), # GraphDef
           '',
           False, # is the GraphDef in binary format
           os.path.join(temp_folder, 'model.ckpt'), # checkpoint name
           output_name, #output node name
           '', '',
           os.path.join(temp_folder, 'deploy.frozen.pb'), # output frozen path graph
           True, # clear devices info from meta-graph
           '', '', '')
def load(temp_folder=None):
    if temp_folder==None:
        temp_folder = os.path.dirname(os.path.realpath(__file__))+'/dest/'
    tf.reset_default_graph()
    #tf_graph = tf.get_default_graph().as_graph_def(add_shapes=True)
    graph = load_graph(os.path.join(temp_folder, 'deploy.frozen.pb'))
    #graph = tf.import_graph_def(os.path.join(temp_folder, 'deploy.frozen.pb'))
    #tf_graph = graph.as_graph_def(add_shapes=True)

    return tf_graph


def load_frozen_graph(temp_folder="dest/"):
    filename = temp_folder + "deploy.frozen.pb"

    with tf.gfile.GFile(filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        #new_input = tf.placeholder(tf.float32, [10], name="new_input")

        tf.import_graph_def(
            graph_def,
            # usually, during training you use queues, but at inference time use placeholders
            # this turns into "input
            #input_map={"input:0": new_input},
            return_elements=None,
            # if input_map is not None, needs a name
            name="bla",
            op_dict=None,
            producer_op_list=None
        )

    #checkpoint_path = tf.train.latest_checkpoint("./tmp/")

    #with tf.Session(graph=graph) as sess:
    #    saver = tf.train.import_meta_graph(checkpoint_path + ".meta", import_scope=None)
    #    saver.restore(sess, checkpoint_path)

    #    output = sess.run("output:0", feed_dict={"input:0": np.arange(10, dtype=np.float32)})
    #    print "output", output

def get_node_by_name(graph, name):
  for node in graph.node:
    print(node.name)
    if node.name == 'prefix/'+name:
      return node

if __name__ == "__main__":
    pr = TF()
    for i in range(10):
        tm = pr.process()
        print(tm)
    print(tm)
