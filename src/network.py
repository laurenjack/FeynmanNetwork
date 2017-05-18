import tensorflow as tf
import numpy as np

class FeedForward:
    """A standard feed-forward neural network, with an operation provided to store each of its
    active linear regions and their properties"""

    def __init__(self, conf):

        self.sizes = conf.sizes

    def train(self):
        pass

    def cost(self):
        pass

    def feed_forward(self):
        #Create place holder for inputs
        in_dim = self.sizes[0]
        x = tf.placeholder(tf.float32, shape=[None, in_dim], name="z_inputs")

        #Create hidden layers
        T = []
        a_out = x
        for s_in, s_out in zip(self.sizes[:-2],self.sizes[1:-1]):
            a_out = self._hidden_layer(s_in, s_out, a_out)
            T.append(tf.greater(a_out, 0), name="make_t")

        #Create softmax layer for classification
        a_out = self._hidden_layer(s_in, s_out, a_out, act_func=tf.nn.softmax)
        return a_out, T

    def _create_layer(self, s_in, s_out, a_in, act_func=tf.nn.relu):
        #name the parameters
        w_name = "W" + str(s_out)
        b_name = "b" + str(s_out)

        #Create parameters
        W = tf.get_variable(w_name, shape=[s_in, s_out], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(0.1, dtype=tf.float32, shape=[s_out], name=b_name)

        #Apply the weighted sum and activation
        z = tf.add(tf.matmul(a_in, W), b)
        a_out = act_func(z)
        return a_out

    def _softmax_layer(self, s_in, s_out, a_in):
        W, b = self._create_weights(s_in, s_out)
        z = tf.add(tf.matmul(a_in, W), b)
        return t

    def _create_weights(self, s_in, s_out):

        return W, b




