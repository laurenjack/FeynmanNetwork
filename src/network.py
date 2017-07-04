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
        l = len(self.sizes) - 2
        self.a_out = self._create_layer(self.sizes[-2], self.sizes[-1], a_out, l, act_func=tf.nn.softmax)

        self.predictions = tf.argmax(self.a_out, axis=1)

        if not self.is_binary:
            #Update magnitude of a_i using path sums of the latter parts of the netwok
            num_to_update = len(self.T[:-1])
            for i in xrange(num_to_update):
                t = self.T[i]
                l = i + 1
                w_name = "W"+str(l)
                w = tf.get_variable(w_name)
                for j in xrange()


        #Create the cost function
        # TODO train for zero too
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.a_out), reduction_indices=[1]))


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
        return tf.reduce_mean(tf.cast(is_correct, tf.float32))

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
        self.weights.append(W)
        self.biases.append(b)

        #Apply the weighted sum and activation
        z = tf.add(tf.matmul(a_in, W), b)
        a_out = act_func(z)
        return a_out

    def _create_path_sums(self):
        num_W_mats = len(self.T[:-1])
        sm_W = self.Ws[-1]
        sm_b = self.bs[-1]
        #For each prediction in the batch, select the weight vector leading into that prediction
        predicted_weights = tf.gather(tf.transpose(sm_W), self.predictions)
        predicted_weights = tf.transpose(predicted_weights)
        predicted_biases = tf.gather(sm_b, self.predictions)

        

        W_mats = []
        for i in xrange(num_W_mats):
            t = self.T[i]
            l = i + 1
            w_name = "W" + str(l)
            w = tf.get_variable(w_name)
            for j in xrange()




