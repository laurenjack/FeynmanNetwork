import tensorflow as tf
import numpy as np
import visualisation as vis
import sys
import cifar
from convnet import Resnet
from reporter import Reporter
from conv_k_nearest import KNearest, KNearestInfo


def run(conf, is_training, global_step):
    train_dir = conf.train_dir
    im_dim = conf.image_dims
    num_classes = conf.num_classes
    k = conf.k
    s = conf.s

    #Create ops
    images = tf.placeholder(tf.float32, shape=[None, im_dim, im_dim, 3], name="image_input")
    targets = tf.placeholder(tf.int32, shape=[None], name="targets")
    resnet = Resnet(conf, is_training, global_step, images, targets)
    rep = Reporter(conf, resnet)
    adv_op = resnet.fgsm_adverserial_example()
    k_nearest = KNearest()
    kn_op = k_nearest.find_k_nearest(k)

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

    #Sample part of validation set to gen adverserial examples and sample part for correct examples
    x, y = train
    x_test, y_test = test
    n = x_test.shape[0]
    rand_inds = np.random.choice(np.arange(n), size=s * 2, replace=False)
    advs_inds = rand_inds[0:s]
    val_inds = rand_inds[s:]

    # Create adverserial examples based on the validation set
    x_adv = x_test[advs_inds]
    y_adv = y_test[advs_inds]
    x_adv = sess.run(adv_op, feed_dict={resnet.is_training: False, images: x_adv, targets: y_adv})

    # Report on the training set
    train_prs = rep.report_full_ds(x, y, sess, n)
    print "Training set reported on"

    # Report on s test set instances
    x_test_samp = x_test[val_inds]
    y_test_samp = y_test[val_inds]
    test_prs = rep.report_full_ds(x_test_samp, y_test_samp, sess, s)

    # Report on the s adverserial examples
    adv_prs = rep.report_full_ds(x_adv, y_adv, sess, s)
    print "Validation and adverserial samples reported on"

    #K nearest
    knrs_validation = _k_nearest_for_each(k_nearest, kn_op, sess, train_prs, test_prs)
    knrs_adverserial = _k_nearest_for_each(k_nearest, kn_op, sess, train_prs, adv_prs)
    print "K Nearest complete"

    #Pass off to visualisation component
    print "Visualising validation samples"
    _visualise_all(knrs_validation)
    print "Visualising adverseries"
    _visualise_all(knrs_adverserial)


def _k_nearest_for_each(k_nearest, kn_op, sess, train_prediction_results, test_prediction_results):
    #unpack training data info
    train_pre_final = train_prediction_results.pre_final
    train_targets = train_prediction_results.targets
    train_predictions = train_prediction_results.predictions
    train_was_correct = train_prediction_results.was_correct
    test_pre_final = test_prediction_results.pre_final
    test_targets = test_prediction_results.targets
    test_predictions = test_prediction_results.predictions

    feed_dict = {k_nearest.train_points: train_pre_final, k_nearest.their_targets: train_targets,
                 k_nearest.their_predictions : train_predictions, k_nearest.train_was_correct: train_was_correct,
                 k_nearest.test_point: None}

    num_test_inst = test_pre_final.shape[0]
    kn_results = []
    #Iterate over test instances, can only feed one by 1 to memory intensive k nearest op
    for i in xrange(num_test_inst):
        feed_dict[k_nearest.test_point] = test_pre_final[i]
        nearest_targets, nearest_predictions, their_distances, were_they_correct, avg_distance =\
            sess.run(kn_op, feed_dict=feed_dict)
        kn_result = KNearestInfo(nearest_targets, nearest_predictions, their_distances, were_they_correct, avg_distance)
        kn_result.target = test_targets[i]
        kn_result.prediction = test_predictions[i]
        kn_results.append(kn_result)
    return kn_results

def _compute_distances(dc, dc_op, train_pre_final, test_pre_final):
    
    feed_dict = {}
    for k in xrange(0, sample_size, self.m):


def _visualise_all(kn_results):
    for knr in kn_results:
        vis.show_neighbouring_instances_conv(knr)




