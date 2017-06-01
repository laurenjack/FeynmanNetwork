import numpy as np
import tensorflow as tf
from linear_regions import *
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
    train_op = network.train()
    feed_forward = network.feed_forward()
    acc_op = network.accuracy()

    #Train network
    for epoch in xrange(epochs):
        np.random.shuffle(batch_indicies)
        for k in xrange(0, n, m):
            batch = batch_indicies[k:k+m]
            x = X[batch]
            y = Y[batch]
            feed_dict = {network.x : x, network.y: y}
            #a = sess.run(feed_forward, feed_dict=feed_dict)
            sess.run(train_op, feed_dict=feed_dict)
        eval_batch = _random_batch(batch_indicies, 1000)
        x = X[eval_batch]
        y = Y[eval_batch]
        feed_dict = {network.x : x, network.y: y}
        acc = sess.run(acc_op, feed_dict=feed_dict)
        print "Epoch "+str(epoch+1)+" Train Acc Sample: "+str(acc)

    #Report on the number of linear regions
    rsb = RegionSetBuilder(conf)
    for k in xrange(0, n, m):
        batch = batch_indicies[k:k + m]
        x = X[batch]
        y = Y[batch]
        feed_dict = {network.x: x, network.y: y}
        a, Ts = sess.run(feed_forward, feed_dict=feed_dict)
        rsb.putRegion(Ts, np.argmax(y, axis=1), np.argmax(a, axis=1))
        print "Sample: "+str(k)+" complete"
    return rsb.get_forest()



def _create_batch_indicies(n):
    return np.arange(n)

def _random_batch(batch_indicies, m):
    return np.random.choice(batch_indicies, size=m, replace=False)