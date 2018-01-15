import numpy as np
import tensorflow as tf
from linear_regions import *
from activation_tuples import *
from network import FeedForward
from conf import Config
import visualisation as vis
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

def report_predictions(trained_network, X, Y, conf, sess):
    """Report the predictions probs on a a given data set"""
    pred_probs_op = trained_network.predictions_probs()
    feed_dict = {trained_network.x: X, trained_network.y: Y}
    probs = sess.run(pred_probs_op, feed_dict=feed_dict)
    m = Y.shape[0]
    for i in xrange(m):
        print "Target: "+str(np.argmax(Y[i]))+"   Predictions: "+str(probs[i])


def gen_adverserial_examples(network, X, Y, conf, sess):
    """Generate an advererial counter-part for each image in X"""
    X = np.copy(X)
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

def to_other_class_adverseries(network, x, y_other, sess, ad_lr= 1.5, thresh=0.9, max_epochs=300):
    """Takes a batch of examples, with true targets y, and creates adverseries of class y_other.
    Each example x[i] is trained using adam, to change the networks prediction from y[i] to y_other[i].
    The training stops when p(y_other[i]) >= thresh."""
    x = np.copy(x)
    adv_grad_op = network.adverserial_rbf_grad() #network.adverserial_grad()
    probs_op = network.predictions_probs()
    m = x.shape[0]
    #batch_ind = np.arange(m)
    #ind_of_relevant_prob = [batch_ind, np.argmax(y_other, axis=1)]
    ind_of_relevant_prob = np.argmax(y_other, axis=1)
    #feed_dict = {network.y: y_other}
    feed_dict = {}
    mask = np.ones(m)
    e = 0
    not_thresh = True
    while not_thresh and e < max_epochs:
        not_thresh = False
        for i in xrange(m):
            feed_dict[network.x] = x[i].reshape(1, 784)
            feed_dict[network.y] = y_other[i].reshape(1, 10)
            grad = sess.run(adv_grad_op, feed_dict=feed_dict)[0]
            probs = sess.run(probs_op, feed_dict)[0]
            #Get the probablilities only for y other
            rel_prob = probs[ind_of_relevant_prob[i]]
            #mask[i] = np.less_equal(rel_prob, thresh).astype(dtype=np.float32)
            #masked_grad = np.sign(grad) * mask.reshape(m, 1)
            under_thresh = np.less_equal(rel_prob, thresh)
            if under_thresh:
                #normed_grad = np.sign(grad)
                normed_grad = grad / (np.sum(grad ** 2.0, axis=0) ** 0.5 + 10.0 ** -30) #.reshape(1, 784) #* mask.reshape(m, 1)
                x[i] -= (ad_lr*normed_grad)
                not_thresh = True
        e += 1
    return x

def gen_black_box_adverseries(num_adv, NET_GLOBAL):
    sizes = [784, 100, 10]
    learning_rate = 0.001
    m = 20
    epochs = 20
    conf = Config(sizes, learning_rate, m, epochs, feyn_lr=0.05, feyn_epochs=100, k=15, epsilon=0.25, is_binary=True,
                  is_w_pixels=False)  # /255.0)
    network = FeedForward(conf, NET_GLOBAL)
    X = conf.X
    Y = conf.Y

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train the network
    train(network, X, Y, m, epochs, sess)

    X_val = conf.X_val
    Y_val = conf.Y_val
    #np.random.shuffle(X_val)
    #np.random.shuffle(Y_val)

    # test the adverserial examples
    #n = X.shape[0]
    val_n = X_val.shape[0]

    rand_inds = np.random.choice(np.arange(val_n), size=num_adv, replace=False)
    X_sub = X_val[rand_inds]
    adv_Y = Y_val[rand_inds]
    # print adv_Y
    # vis.show_original_vs_adv(X_sub, X_sub)
    adv_X = gen_adverserial_examples(network, X_sub, adv_Y, conf, sess)
    # print adv_Y
    # vis.show_original_vs_adv(X_sub, adv_X)
    return adv_X, X_sub, adv_Y



def run_k_nearest():
    pass

def _create_batch_indicies(n):
    return np.arange(n)

def _random_batch(batch_indicies, m):
    return np.random.choice(batch_indicies, size=m, replace=False)

from activation_tuples import TupleSetBuilder

def create_rsb(conf):
    # if conf.is_binary:
    #     return RegionSetBuilder(conf)
    return TupleSetBuilder(conf)

