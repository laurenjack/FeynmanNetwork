import numpy as np
import tensorflow as tf
from linear_regions import *
from activation_tuples import *
#from tensorflow.python import debug as tf_debug

def train_and_report(network, X, Y, conf):
    """Train network with a training set specifed by:

    :param X - All inputs in the training set

    :param Y - All targets corresponding to the inputs in the training set

    :return network, fully trained
    """
    n = X.shape[0]
    m = conf.m
    epochs = conf.epochs

    batch_indicies = _create_batch_indicies(n)
    sess = tf.InteractiveSession()
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    tf.global_variables_initializer().run()
    #train_op = network.train()
    feed_forward = network.feed_forward()
    #acc_op = network.accuracy()

    #Train network
    train(network, X, Y, m, epochs, sess)

    #Report on the number of linear regions
    rsb = create_rsb(conf)
    print "Building regions ..."
    for k in xrange(0, n, m):
        batch = batch_indicies[k:k + m]
        x = X[batch]
        y = Y[batch]
        feed_dict = {network.x: x, network.y: y}
        a, Ts = sess.run(feed_forward, feed_dict=feed_dict)
        rsb.putRegion(Ts, np.argmax(y, axis=1), np.argmax(a, axis=1))
    print "Region Building Complete."
    return rsb.get_forest(), sess



def train(network, X, Y, m, epochs, sess, sub_set_report=1000, is_feyn=False):
    n = X.shape[0]
    batch_indicies = _create_batch_indicies(n)

    train_op = network.train()
    acc_op = network.accuracy()
    wb = network.wb()

    # Train network
    for epoch in xrange(epochs):
        np.random.shuffle(batch_indicies)
        for k in xrange(0, n, m):
            batch = batch_indicies[k:k + m]
            x = X[batch]
            y = Y[batch]
            feed_dict = {network.x: x, network.y: y}
            # a = sess.run(feed_forward, feed_dict=feed_dict)
            sess.run(train_op, feed_dict=feed_dict)
            if is_feyn:
                w, b = sess.run(wb, feed_dict=feed_dict)
                print "W: "+str(w)+"b: "+str(b)
        eval_batch = _random_batch(batch_indicies, sub_set_report)
        x = X[eval_batch]
        y = Y[eval_batch]
        feed_dict = {network.x: x, network.y: y}
        acc, z, a = sess.run(acc_op, feed_dict=feed_dict)
        print "Epoch " + str(epoch + 1) + " Train Acc Sample: " + str(acc)


def report(trained_network, X, Y, conf, sess):
    # Report on the number of linear regions
    n = X.shape[0]
    m = conf.m
    feed_forward = trained_network.feed_forward()
    batch_indicies = _create_batch_indicies(n)
    rsb = create_rsb(conf)
    print "Building regions ..."
    for k in xrange(0, n, m):
        batch = batch_indicies[k:k + m]
        x = X[batch]
        y = Y[batch]
        feed_dict = {trained_network.x: x, trained_network.y: y}
        a, Ts = sess.run(feed_forward, feed_dict=feed_dict)
        rsb.putRegion(Ts, np.argmax(y, axis=1), np.argmax(a, axis=1))
    print "Region Building Complete."
    return rsb.get_forest()

def gen_adverserial_examples(network, X, Y, conf, sess):
    """Generate an advererial counter-part for each image in X"""
    n = X.shape[0]
    m = conf.m
    adverserial_op = network.fgsm_adverserial_example()
    batch_indicies = _create_batch_indicies(n)
    ad_xs = []
    for k in xrange(0, n, m):
        batch = batch_indicies[k:k + m]
        x = X[batch]
        y = Y[batch]
        feed_dict = {network.x: x, network.y: y}
        a, b, c, ad_x = sess.run(adverserial_op, feed_dict=feed_dict)
        ad_xs.append(ad_x)
    return np.concatenate(ad_xs)


def run_k_nearest():
    pass

def _create_batch_indicies(n):
    return np.arange(n)

def _random_batch(batch_indicies, m):
    return np.random.choice(batch_indicies, size=m, replace=False)

from activation_tuples import TupleSetBuilder

def create_rsb(conf):
    if conf.is_binary:
        return RegionSetBuilder(conf)
    return TupleSetBuilder(conf)

