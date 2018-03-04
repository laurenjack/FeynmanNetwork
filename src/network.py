import tensorflow as tf
import numpy as np


class FeedForward:
    """A standard feed-forward neural network, with an operation provided to store each of its
    active linear regions and their properties"""

    def __init__(self, conf, NET_GLOBAL):
        self.NET_GLOBAL = NET_GLOBAL
        self.sizes = conf.sizes
        self.learning_rate = conf.learning_rate
        self.epsilon = conf.epsilon
        self.num_h_layers = len(self.sizes) - 2
        self.is_binary = conf.is_binary
        self.is_w_pixels = conf.is_w_pixels
        in_dim = self.sizes[0]
        self.out_dim = self.sizes[-1]
        # Create place holder for inputs and target outputs
        self.x = tf.placeholder(tf.float32, shape=[None, in_dim], name="inputs")
        self.y = tf.placeholder(tf.float32, shape=[None, self.out_dim], name="target_outputs")

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
        if conf.is_rbf:
            self.a_out = self._rbf_final_layer(self.sizes[-2], self.sizes[-1], a_out, l)
        else:
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
        self.rbf_entro = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.a_out), reduction_indices=[1]))
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.a_out + 10 ** -10.0), reduction_indices=[1])) #+ (1.0 - self.y) * tf.log(1.0 - self.a_out + 10 ** -50.0)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)
        self.train_op = self.opt.minimize(self.cross_entropy)
        self.NET_GLOBAL.inc()


    def feed_forward(self):
        return self.a_out, self.T

    def train(self):
        #train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cross_entropy)
        return self.train_op

    def predictions_probs(self):
        return self.a_out

    def fgsm_adverserial_example(self):
        """Generate an adverserial example according to the fast
        gradient sign method (Goodfellow 2015)"""
        image_grads = self.adverserial_grad()
        grad_signs = tf.sign(image_grads)
        pertubation = tf.multiply(self.epsilon, grad_signs)
        return image_grads, grad_signs, pertubation, tf.add(self.x, pertubation)

    def adverserial_grad(self):
        return tf.gradients(self.cross_entropy, self.x)[0]

    def adverserial_rbf_grad(self):
        return tf.gradients(self.rbf_entro, self.x)[0]

    # def adverserial_train_to_other(self):
    #     """Used to train a batch of inputs to match another target"""
    #     batch_inds = tf.reshape(tf.range(self.out_dim), [self.out_dim, 1])
    #     rel_inds = tf.concatenate([batch_inds, tf.reshape(self.y, [self.out_dim, 1]))
    #     grad = self.adverserial_grad()
    #     rel_probs = tfself.a_out
    #     probs = sess.run(probs_op, feed_dict)
    #     # Get the probablilities only for y other
    #     rel_probs = probs[ind_of_relevant_prob]
    #     mask = np.greater(rel_probs, thresh).astype(dtype=np.float32)


    def accuracy(self):
        a, _ = self.feed_forward()
        is_correct = tf.equal(tf.argmax(self.y, 1), tf.argmax(a, 1))
        return tf.reduce_mean(tf.cast(is_correct, tf.float32)), self.z, self.a_out

    def wb(self):
        return self.Ws[0], self.bs[0], self.x_bar

    def get_inaccurate_indicies(self):
        a, T = self.feed_forward()
        predictions = tf.argmax(a, axis=1)
        actuals = tf.argmax(self.y, axis=1)


    def make_prediction(self):
        return self.predictions, self.T

    def _create_layer(self, s_in, s_out, a_in, l, act_func=tf.nn.relu):
        #name the parameters
        w_name = "W" + str(l)+"_"+str(self.NET_GLOBAL.count)
        b_name = "b" + str(l)+"_"+str(self.NET_GLOBAL.count)

        #Create parameters
        W = tf.get_variable(w_name, shape=[s_in, s_out], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(0.1 * np.ones(s_out), dtype=tf.float32, name=b_name)
        self.Ws.append(W)
        self.bs.append(b)

        #Apply the weighted sum and activation
        z = tf.add(tf.matmul(a_in, W), b)
        if act_func is not None:
            z = act_func(z)
        return z

    def _rbf_final_layer(self, s_in, s_out, a_in, l):
        lin_layer = self._create_layer(s_in, s_in, a_in, l, act_func=None)
        w_name = "W" + str(l+1)+"_"+str(self.NET_GLOBAL.count)
        W = tf.abs(tf.get_variable(w_name, shape=[s_in, s_out], initializer=tf.contrib.layers.xavier_initializer())) ** 0.5
        x_bar = tf.get_variable('x_bar', shape=[s_in, s_out], initializer=tf.contrib.layers.xavier_initializer())
        x_diff_sq = tf.square(tf.multiply(tf.reshape(W ** 2.0, [1, s_in, s_out]), tf.reshape(lin_layer, [-1, s_in, 1])) - tf.reshape(x_bar, [1, s_in, s_out]))
        dist = tf.reduce_sum(x_diff_sq, axis=1)
        self.rbf = 10.0 * tf.exp(-dist)
        self.Ws.append(W)
        self.x_bar = x_bar
        self.rbf_W = W
        return tf.nn.softmax(self.rbf)

    def max_rbf(self):
        return self.rbf_entro


    def get_final_layer_params(self):
        return self.rbf_W, self.x_bar

    def get_pre_rbf(self):
        return self.z

    # def _bn(self, x):
    #     x_shape = x.get_shape()
    #     params_shape = x_shape[-1:]
    #     axis = list(range(len(x_shape) - 1))
    #
    #     beta = self._get_variable('beta',
    #                               params_shape,
    #                               initializer=tf.zeros_initializer)
    #     gamma = self._get_variable('gamma',
    #                                params_shape,
    #                                initializer=tf.ones_initializer)
    #
    #     moving_mean = self._get_variable('moving_mean',
    #                                      params_shape,
    #                                      initializer=tf.zeros_initializer,
    #                                      trainable=False)
    #     moving_variance = self._get_variable('moving_variance',
    #                                          params_shape,
    #                                          initializer=tf.ones_initializer,
    #                                          trainable=False)
    #
    #     # These ops will only be preformed when training.
    #     mean, variance = tf.nn.moments(x, axis)
    #     update_moving_mean = moving_averages.assign_moving_average(moving_mean,
    #                                                                mean, BN_DECAY)
    #     update_moving_variance = moving_averages.assign_moving_average(
    #         moving_variance, variance, BN_DECAY)
    #     tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    #     tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
    #
    #     mean, variance = control_flow_ops.cond(
    #         self.is_training, lambda: (mean, variance),
    #         lambda: (moving_mean, moving_variance))
    #
    #     x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    #     # x.set_shape(inputs.get_shape()) ??
    #
    #     return x

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






