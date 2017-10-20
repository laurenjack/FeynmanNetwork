import tensorflow as tf
import numpy as np

class FeedForward:
    """A standard feed-forward neural network, with an operation provided to store each of its
    active linear regions and their properties"""

    def __init__(self, conf):
        self.sizes = conf.sizes
        self.learning_rate = conf.learning_rate
        self.epsilon = conf.epsilon
        self.num_h_layers = len(self.sizes) - 2
        self.is_binary = conf.is_binary
        self.is_w_pixels = conf.is_w_pixels
        in_dim = self.sizes[0]
        out_dim = self.sizes[-1]
        # Create place holder for inputs and target outputs
        self.x = tf.placeholder(tf.float32, shape=[None, in_dim], name="inputs")
        self.y = tf.placeholder(tf.float32, shape=[None, out_dim], name="target_outputs")

        #Create the network
        self.Ws = []
        self.bs = []
        self.T = []
        a_out = self.x
        for s_in, s_out, l in zip(self.sizes[:-2], self.sizes[1:-1], xrange(self.num_h_layers)):
            a_out = self._create_layer(s_in, s_out, a_out, l)
            t = a_out
            if self.is_binary:
                t = tf.greater(a_out, 0)
            self.T.append(t)

        # Create softmax layer for classification
        self.z = a_out
        l = len(self.sizes) - 2
        self.a_out = self._create_layer(self.sizes[-2], self.sizes[-1], a_out, l, act_func=tf.nn.softmax)

        with tf.device("/cpu:0"):
            self.predictions = tf.argmax(self.a_out, axis=1)
            self.targets = tf.argmax(self.y, axis=1)
            self.predictions = tf.cast(self.predictions, tf.int32)
            pred_indices = tf.reshape(self.predictions, shape=[-1, 1])
            m = tf.shape(self.predictions)[0]
            b_indices = tf.range(0, m)
            b_indices = tf.reshape(b_indices, shape=[-1, 1])
            indices = tf.concat([b_indices, pred_indices], axis=1)
            zeros = tf.zeros(tf.shape(self.predictions))
            max_removed = tf.Variable(initial_value=tf.zeros(shape=[5,5]), trainable=False, validate_shape=False)
            max_removed = tf.assign(max_removed, self.a_out, validate_shape=False)
            max_removed = tf.scatter_nd_update(max_removed, indices, zeros)
            self.second_predictions = tf.argmax(max_removed, axis=1)

        if not self.is_binary:
            self.scale_list = []
            self._scale_regions()


        #Create the cost function
        # TODO train for zero too
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.a_out) + (1.0 - self.y) * tf.log(1.0 - self.a_out + 10 ** -10.0), reduction_indices=[1]))


    def feed_forward(self):
        return self.a_out, self.T

    def train(self):
        train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cross_entropy)
        return train_op

    def fgsm_adverserial_example(self):
        """Generate an adverserial example according to the fast
        gradient sign method (Goodfellow 2015)"""
        image_grads = tf.gradients(self.cross_entropy, self.x)[0]
        grad_signs = tf.sign(image_grads)
        pertubation = tf.multiply(self.epsilon, grad_signs)
        return image_grads, grad_signs, pertubation, tf.add(self.x, pertubation)


    def accuracy(self):
        a, _ = self.feed_forward()
        is_correct = tf.equal(tf.argmax(self.y, 1), tf.argmax(a, 1))
        return tf.reduce_mean(tf.cast(is_correct, tf.float32)), self.z, self.a_out

    def wb(self):
        return None, None

    def get_inaccurate_indicies(self):
        a, T = self.feed_forward()
        predictions = tf.argmax(a, axis=1)
        actuals = tf.argmax(self.y, axis=1)


    def make_prediction(self):
        return self.predictions, self.T

    def _create_layer(self, s_in, s_out, a_in, l, act_func=tf.nn.relu):
        #name the parameters
        w_name = "W" + str(l)
        b_name = "b" + str(l)

        #Create parameters
        W = tf.get_variable(w_name, shape=[s_in, s_out], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(0.1 * np.ones(s_out), dtype=tf.float32, name=b_name)
        self.Ws.append(W)
        self.bs.append(b)

        #Apply the weighted sum and activation
        z = tf.add(tf.matmul(a_in, W), b)
        a_out = act_func(z)
        return a_out

    def _scale_regions(self):
        sm_W = self.Ws[-1]
        # For each prediction in the batch, select the weight vector leading into that prediction
        #self.all_same = 4 * tf.ones(tf.shape(self.predictions), dtype=tf.int32)
        predicted_weights = tf.gather(tf.transpose(sm_W), self.predictions)
        #scales = predicted_weights
        scales = tf.ones(tf.shape(predicted_weights))
        #scales = tf.ones(tf.shape(self.y))
        #scales = tf.transpose(tf.matmul(sm_W, tf.transpose(scales)))
        bias_sum = tf.expand_dims(tf.zeros(tf.shape(tf.reduce_sum(self.y, axis=1))), axis=1)

        for l in xrange(self.num_h_layers - 1, -1, -1):
            self.scale_list.append(scales)
            a = self.T[l]
            l_size = self.sizes[l]
            self.T[l] = tf.multiply(a, scales) + bias_sum
            bias_sum += tf.matmul(scales, tf.expand_dims(self.bs[l], axis=1))
            # Remove all paths which do not influence the output as they feed from a zero
            scales = tf.where(tf.greater(a, 0), tf.zeros(shape=tf.shape(scales)), scales)
            scales = tf.transpose(tf.matmul(self.Ws[l], tf.transpose(scales)))

        #Weight the pixels now
        if self.is_w_pixels:
            self.scale_list.append(scales)
            self.T = tf.multiply(tf.ones(tf.shape(self.x)), scales) + bias_sum
        else:
            self.scale_list.append(scales)
            self.T.append(tf.multiply(self.x, scales) + bias_sum)

        self.T = [self.T[-2]]






