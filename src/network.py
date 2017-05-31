import tensorflow as tf
import numpy as np

class FeedForward:
    """A standard feed-forward neural network, with an operation provided to store each of its
    active linear regions and their properties"""

    def __init__(self, conf):
        self.sizes = conf.sizes
        self.learning_rate = conf.learning_rate
        self.num_h_layers = len(self.sizes) - 2
        in_dim = self.sizes[0]
        out_dim = self.sizes[-1]
        # Create place holder for inputs and target outputs
        self.x = tf.placeholder(tf.float32, shape=[None, in_dim], name="inputs")
        self.y = tf.placeholder(tf.float32, shape=[None, out_dim], name="target_outputs")

        #Create the network
        self.T = []
        a_out = self.x
        for s_in, s_out, l in zip(self.sizes[:-2], self.sizes[1:-1], xrange(self.num_h_layers)):
            a_out = self._create_layer(s_in, s_out, a_out, l)
            self.T.append(tf.greater(a_out, 0))

        # Create softmax layer for classification
        l = len(self.sizes) - 2
        self.a_out = self._create_layer(self.sizes[-2], self.sizes[-1], a_out, l, act_func=tf.nn.softmax)

    def feed_forward(self):
        return self.a_out, self.T

    def train(self):
        a, _ = self.feed_forward()
        #TODO train for zero too
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(a), reduction_indices=[1]))
        train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cross_entropy)
        return train_op

    def accuracy(self):
        a, _ = self.feed_forward()
        is_correct = tf.equal(tf.argmax(self.y, 1), tf.argmax(a, 1))
        return tf.reduce_mean(tf.cast(is_correct, tf.float32))

    def get_inaccurate_indicies(self):
        a, T = self.feed_forward()
        predictions = tf.argmax(a, axis=1)
        actuals = tf.argmax(self.y, axis=1)


    def make_prediction(self):
        a, T = self.feed_forward()
        predictions = tf.argmax(a, axis=1)
        return predictions, T

    def _create_layer(self, s_in, s_out, a_in, l, act_func=tf.nn.relu):
        #name the parameters
        w_name = "W" + str(l)
        b_name = "b" + str(l)

        #Create parameters
        W = tf.get_variable(w_name, shape=[s_in, s_out], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(0.1 * np.ones(s_out), dtype=tf.float32, name=b_name)

        #Apply the weighted sum and activation
        z = tf.add(tf.matmul(a_in, W), b)
        a_out = act_func(z)
        return a_out




