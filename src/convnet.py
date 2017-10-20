import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from reporter import PredictionResult


MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0 #0.0001
#CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0
#FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op
IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]

class Resnet:




    def __init__(self, conf, is_training, global_step, images, labels):
        #The network
        self.num_classes = conf.num_classes
        self.image_dims = conf.image_dims
        self.stacks = conf.stacks

        # Generating adverserial examples
        self.adv_epsilon = conf.adv_epsilon

        self.pp_k_size = conf.pp_k_size
        self.pp_stride = conf.pp_stride
        self.pp_d = conf.pp_d
        self.has_mp = conf.has_mp

        self.is_training = is_training
        self.global_step = global_step
        self.images = images
        self.labels = labels

        # Build the network
        # Construct pre pooling layer
        self.act_tups = []
        a = self._layer(images, self.pp_d, self.pp_k_size, self.pp_stride, 'pre_pool')
        # apply max pooling
        if self.has_mp:
            a = self._max_pool(a)

        num_stacks = len(self.stacks)
        for stack, i in zip(self.stacks, range(num_stacks)):
            stack_scope = 'stack' + str(i)
            a = self._stack(a, stack, stack_scope)

        # self.pre_final = a
        a = tf.reduce_mean(a, reduction_indices=[1, 2], name="avg_pool")
        #self.pre_final = tf.multiply(self.pre_final, tf.reshape(a, [-1,1,1,64]))
        #self.pre_final = tf.contrib.layers.flatten(self.pre_final)
        self.act_tups = tf.concat(self.act_tups, axis=1)

        with tf.variable_scope('fc'):
            self.logits = self._fc(a)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
        self.sm_loss = tf.reduce_mean(cross_entropy)

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        decay_lr = tf.train.exponential_decay(0.0002, global_step, 480000, 0.2, True)
        reg_losses = [tf.multiply(decay_lr, reg_loss) for reg_loss in reg_losses]

        self.loss = tf.add_n([self.sm_loss] + reg_losses)
        tf.summary.scalar('loss', self.loss)


    def fgsm_adverserial_example(self):
        """Generate an adverserial example according to the fast
        gradient sign method (Goodfellow 2015)"""
        image_grads = tf.gradients(self.sm_loss, self.images)[0]
        grad_signs = tf.sign(image_grads)
        pertubation = tf.multiply(self.adv_epsilon, grad_signs)
        return tf.add(self.images, pertubation)

    def inference(self):
        return self.logits

    def prediction_result(self):
        vec_predictions = tf.nn.softmax(self.logits)
        predictions = tf.argmax(vec_predictions, axis=1, output_type=tf.int32)
        was_correct = tf.equal(predictions, self.labels)
        return self.act_tups, predictions, self.labels, was_correct

    def inference_and_pre_final(self):
        return self.logits, self.act_tups

    def just_sm_loss(self):
        return self.sm_loss

    def loss(self):
        return self.loss


    def _stack(self, a, stack, scope_name):
        with tf.variable_scope(scope_name):
            # transition to this stack, by creating the first block
            f_stride = stack.f_stride
            in_d = stack.in_d
            if scope_name == 'stack0':
                a = self._block(a, stack, "block0")
            else:
                with tf.variable_scope("block0"):
                    skip = tf.layers.average_pooling2d(a, 2, 2)  # self._layer(a, stack.in_d, 1, 2, "skip_projection")
                    prev_out = tf.shape(a)[3]
                    skip = tf.pad(skip, [[0, 0], [0, 0], [0, 0], [prev_out // 2, prev_out // 2]])
                    a = self._layer(a, in_d, stack.k_size, f_stride, "layer_0")
                    a = self._layer(a, in_d, stack.k_size, stack.stride, "layer_1", skip=skip)
            for b in xrange(stack.num_blocks-1):
                block_scope = "block"+str(b+1)
                a = self._block(a, stack, block_scope)
            return a

    def _block(self, a, stack, scope_name):
        k_size = stack.k_size
        stride = stack.stride
        in_d = stack.in_d
        with tf.variable_scope(scope_name):
            skip = a
            a = self._layer(a, in_d, k_size, stride, "layer_0")
            a = self._layer(a, in_d, k_size, stride, "layer_1", skip=skip)
            return a

    def _layer(self, a, out_d, k_size, stride, scope_name, skip=None):
        with tf.variable_scope(scope_name):
            a = self._conv(a, out_d, k_size, stride)
            a = self._bn(a)
            if skip is not None:
                a = a + skip
        a = tf.nn.relu(a)
        act_tup = tf.greater(a, 0.0)
        act_tup = tf.contrib.layers.flatten(act_tup)
        self.act_tups.append(act_tup)
        return a

    def _conv(self, x, filters_out, k_size, stride):
        filters_in = x.get_shape()[-1]
        shape = [k_size, k_size, filters_in, filters_out]
        initializer = tf.contrib.layers.variance_scaling_initializer(mode='FAN_OUT') # tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
        weights = self._get_variable('weights',
                                shape=shape,
                                initializer=initializer,
                                weight_decay=CONV_WEIGHT_DECAY)
        return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')

    def _max_pool(self, x, ksize=3, stride=2):
        return tf.nn.max_pool(x,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, stride, stride, 1],
                              padding='SAME')

    def _bn(self, x):
        x_shape = x.get_shape()
        params_shape = x_shape[-1:]
        axis = list(range(len(x_shape) - 1))

        beta = self._get_variable('beta',
                             params_shape,
                             initializer=tf.zeros_initializer)
        gamma = self._get_variable('gamma',
                              params_shape,
                              initializer=tf.ones_initializer)

        moving_mean = self._get_variable('moving_mean',
                                    params_shape,
                                    initializer=tf.zeros_initializer,
                                    trainable=False)
        moving_variance = self._get_variable('moving_variance',
                                        params_shape,
                                        initializer=tf.ones_initializer,
                                        trainable=False)

        # These ops will only be preformed when training.
        mean, variance = tf.nn.moments(x, axis)
        update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                                   mean, BN_DECAY)
        update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, variance, BN_DECAY)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

        mean, variance = control_flow_ops.cond(
            self.is_training, lambda: (mean, variance),
            lambda: (moving_mean, moving_variance))

        x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
        # x.set_shape(inputs.get_shape()) ??

        return x

    def _fc(self, a):
        num_units_in = a.get_shape()[1]
        num_units_out = self.num_classes
        weights_initializer = tf.contrib.layers.variance_scaling_initializer()

        weights = self._get_variable('weights',
                                shape=[num_units_in, num_units_out],
                                initializer=weights_initializer,
                                weight_decay=FC_WEIGHT_DECAY)
        biases = self._get_variable('biases',
                               shape=[num_units_out],
                               initializer=tf.zeros_initializer)
        a = tf.nn.xw_plus_b(a, weights, biases)
        weight_reg = tf.nn.l2_loss(weights)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_reg)
        return a

    def _get_variable(self, name,
                      shape,
                      initializer,
                      weight_decay=0.0,
                      dtype=tf.float32,
                      trainable=True):
        "A little wrapper around tf.get_variable to do weight decay and add to"
        "resnet collection"
        # if weight_decay > 0:
        #     regularizer = #tf.contrib.layers.l2_regularizer(weight_decay)
        # else:
        #     regularizer = None
        collections = [tf.GraphKeys.GLOBAL_VARIABLES, RESNET_VARIABLES]
        return tf.get_variable(name,
                               shape=shape,
                               initializer=initializer,
                               dtype=dtype,
                               regularizer=None,
                               collections=collections,
                               trainable=trainable)