import numpy as np
import math
import tensorflow as tf
from resnet import Resnet
import sys
import cifar

def x_bar_vs_centre_resnet(conf, is_training, global_step, x_bar):
    train_dir = conf.train_dir
    im_dim = conf.image_dims
    num_classes = conf.num_classes
    k = conf.k
    s = conf.s
    m = conf.reporting_m

    # Create ops
    images = tf.placeholder(tf.float32, shape=[None, im_dim, im_dim, 3], name="image_input")
    targets = tf.placeholder(tf.int32, shape=[None], name="targets")
    resnet = Resnet(conf, is_training, global_step, images, targets)
    pre_rbf = resnet.z

    # Tensorflow admin
    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    # Load network
    latest = tf.train.latest_checkpoint(train_dir)
    if not latest:
        print "No checkpoint to continue from in", train_dir
        sys.exit(1)
    print "resume", latest
    saver.restore(sess, latest)

    # Load Cifar 10, training and test set, images and labels
    cifar10 = cifar.Cifar10()
    train, test = cifar10.load_full_np()
    X, Y = train
    n = X.shape[0]

    #Iterate through the training set once, calcuating the centre for each class
    batch_inds = np.arange(n)

    centre_dict = {t: np.zeros(resnet.z_d) for t in range(10)}
    count_dict = {t:0.0 for t in range(10)}
    for i in xrange(0, n, m):
        inds = batch_inds[i:i+m]
        x = X[inds]
        y = Y[inds]
        pre_rbf_batch = sess.run(pre_rbf, {images: x, targets: y, is_training: False})
        for j in xrange(m):
            t = y[j]
            centre_dict[t] += pre_rbf_batch[j]
            count_dict[t] += 1.0
    #Retrieve the x_bars and taus
    x_bar, W = sess.run([resnet.x_bar, resnet.W], {is_training: False})
    means = np.array([centre_dict[t] / count_dict[t] for t in range(10)])
    mean_diff_xbar = means - x_bar.transpose()
    print "MEANS"
    print means
    print "MEAN DIFFS"
    print np.sum(mean_diff_xbar ** 2.0, axis=1) ** 0.5 / 64.0




def x_bar_vs_centre(X, Y, x_bar, W, sess, network):
    # Get indicies of each class
    bin_targets = np.argmax(Y, axis=1)
    indicies_of_5 = np.argwhere(bin_targets == 5)[:, 0]
    x_bar_5 = x_bar[:, 5]
    w5 = W[:, 5]
    #Run feed forward to get the z points of the 5's
    pre_rbf_op = network.get_pre_rbf()
    just_5s = sess.run(pre_rbf_op, feed_dict={network.x: X[indicies_of_5], network.y: Y[indicies_of_5]})
    centre = np.mean(just_5s, axis=0)
    dist_x_bar_centre = np.sum((x_bar_5 - centre) ** 2.0) ** 0.5
    w_dist_x_bar_centre = np.sum((w5*(x_bar_5 - centre)) ** 2.0) ** 0.5 / x_bar_5.shape[0]
    three_dists = np.sum((just_5s[0:3] - x_bar_5) ** 2.0, axis=1) ** 0.5
    w_three_dists = np.sum((w5 * (just_5s[0:3] - x_bar_5)) ** 2.0, axis=1) ** 0.5
    print "Average dist: "+str(dist_x_bar_centre)
    print "Three dists: "+str(three_dists)
    print "Average Weighted dist: " + str(w_dist_x_bar_centre)
    print "Three Weighted dists: " + str(w_three_dists)



def find_pairwise_intersections(network, epochs, lr, lmda, sess):
    """Given an Neural network with an rbf-softmax final layer, report the maximum point of intersection between
    each pair of rbfs"""
    W, xbar = network.get_final_layer_params()
    #Convert to numpy arrays
    W = sess.run(W)
    xbar = sess.run(xbar)
    #Select the pairs of clusters
    w1 = W[:,1]
    w7 = W[:,7]
    bar1 = xbar[:,1]
    bar7 = xbar[:,7]

    #Initialize the x as the centre point, run sgd
    x = (bar1 + bar7) / 2.0
    for e in xrange(epochs):
        grad, con = compute_grad_and_constraint(w1, w7, bar1, bar7, x)
        #Normalise each
        #grad /= np.sum(grad ** 2.0) ** 0.5
        #con /= np.sum(con ** 2.0) ** 0.5
        x -= lr * grad + lmda * con
        rbf1 = normed_rbf(w1, bar1, x)
        rbf7 = normed_rbf(w7, bar7, x)

    rbf1 = normed_rbf(w1, bar1, x)
    rbf7 = normed_rbf(w7, bar7, x)
    # print bar1
    # print bar7
    # print w1
    # print w7
    return rbf1, rbf7



def compute_grad_and_constraint(w1, w2, x_bar1, x_bar2, x):
    d1 = w1 ** 2.0 * (x - x_bar1)
    d2 = w2 ** 2.0 * (x - x_bar2)
    grad = d1 + d2
    constraint = d1 - d2
    return grad, constraint

def normed_rbf(w, x_bar, x):
    scale = 1.0 # (1.0 / (2 * math.pi * np.sum(1.0 / w ** 2.0))) ** 0.5
    exp = np.exp(-0.5 * np.sum((w * (x - x_bar)) ** 2.0))
    return scale * exp