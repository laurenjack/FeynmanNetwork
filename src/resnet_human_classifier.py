import tensorflow as tf
import numpy as np
import visualisation as vis
import sys
import cifar
from resnet import Resnet
from reporter import Reporter
from conv_k_nearest import DistanceComputer, KNearest, TrainCounter, KNearestInfo


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
    dc = DistanceComputer()
    dc_op = dc.distances()
    k_nearest = KNearest()
    kn_op = k_nearest.find_k_nearest(k)
    tc = TrainCounter(k, 1136, num_classes)
    tc_op = tc.find_count_dist()

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
    n = x.shape[0]
    n_test = x_test.shape[0]
    rand_inds = np.random.choice(np.arange(n_test), size=s * 2, replace=False)
    advs_inds = rand_inds[0:s]
    val_inds = rand_inds[s:]

    # Create adverserial examples based on the validation set
    x_adv = x_test[advs_inds]
    y_adv = y_test[advs_inds]
    x_adv = sess.run(adv_op, feed_dict={resnet.is_training: False, images: x_adv, targets: y_adv})

    # Report on the training set
    train_prs, _= rep.report_full_ds(x, y, sess, n)
    print "Training set reported on"

    # Report on s test set instances
    x_test_samp = x_test[val_inds]
    y_test_samp = y_test[val_inds]
    test_prs, _ = rep.report_full_ds(x_test_samp, y_test_samp, sess, s)

    # Report on the s adverserial examples
    adv_prs, _ = rep.report_full_ds(x_adv, y_adv, sess, s)
    print "Validation and adverserial samples reported on"

    #Create sets by class on training set
    class_organised = _form_class_cluster(train_prs, num_classes)

    #K nearest
    knrs_validation, tc_val = _k_nearest_for_each(k_nearest, kn_op, dc, dc_op, sess, train_prs, test_prs, tc, tc_op, class_organised) #ftws)
    knrs_adverserial, tc_adv = _k_nearest_for_each(k_nearest, kn_op, dc, dc_op, sess, train_prs, adv_prs, tc, tc_op, class_organised) #ftws)
    print "K Nearest complete"

    #Pass off to visualisation component
    print "Visualising validation samples"
    _visualise_all(knrs_validation, tc_val)
    print "Visualising adverseries"
    _visualise_all(knrs_adverserial, tc_adv)


def _k_nearest_for_each(k_nearest, kn_op, dc, dc_op, sess, train_prediction_results, test_prediction_results, tc, tc_op, co, ftws=None):
    #unpack training data info
    train_pre_final = train_prediction_results.pre_final
    train_targets = train_prediction_results.targets
    train_predictions = train_prediction_results.predictions
    train_was_correct = train_prediction_results.was_correct
    test_pre_final = test_prediction_results.pre_final
    test_targets = test_prediction_results.targets
    test_predictions = test_prediction_results.predictions

    feed_dict = {k_nearest.distances: None, k_nearest.their_targets: train_targets,
                 k_nearest.their_predictions : train_predictions, k_nearest.train_was_correct: train_was_correct}

    fd = {}
    tc_fd = {t: cc for t, cc in zip(tc.class_clusters, co.act_tups)}
    num_test_inst = test_pre_final.shape[0]
    kn_results = []
    cc_results = []
    #Iterate over test instances, can only feed one by 1 to memory intensive k nearest op
    for i in xrange(num_test_inst):
        #Compute K nearest for each class
        K_nearest_neighbours = []
        # for j in xrange(10):
        #     act_tups = co.act_tups[j]
        #     fd[k_nearest.distances] = _compute_distances(dc, dc_op, act_tups, test_pre_final[i], sess)
        #     fd[k_nearest.their_targets] = co.targets[j]
        #     fd[k_nearest.their_predictions] = co.predictions[j]
        #     fd[k_nearest.train_was_correct] = co.was_correct[j]
        #     _, _, _, _, _, indicies = sess.run(kn_op, feed_dict=fd)
        #     K_nearest_neighbours.append(act_tups[indicies])

        #tc_fd = {t: cc for t, cc in zip(tc.class_clusters, K_nearest_neighbours)}
        tc_fd[tc.test_point] = test_pre_final[i]
        cc_dists = sess.run(tc_op, feed_dict=tc_fd)
        cc_results.append(cc_dists)
        feed_dict[k_nearest.distances] = _compute_distances(dc, dc_op, train_pre_final, test_pre_final[i], sess, ftws)
        nearest_targets, nearest_predictions, their_distances, were_they_correct, avg_distance, _ =\
            sess.run(kn_op, feed_dict=feed_dict)
        kn_result = KNearestInfo(nearest_targets, nearest_predictions, their_distances, were_they_correct, avg_distance)
        kn_result.target = test_targets[i]
        kn_result.prediction = test_predictions[i]
        kn_results.append(kn_result)
        print 'KNearest '+str(i)
    return kn_results, cc_results


def _compute_distances(dc, dc_op, train_pre_final, test_pre_final, sess, ftws=None):
    train_shape = train_pre_final.shape
    sample_size = train_shape[0]
    distances = np.zeros(sample_size)
    feed_dict = {dc.train_points: None, dc.test_point: test_pre_final}
    for k in xrange(0, sample_size, sample_size):
        feed_dict[dc.train_points] = train_pre_final[k:k+sample_size]
        #feed_dict[dc.ftws] = ftws[k:k+sample_size]
        distances[k:k+sample_size] = sess.run(dc_op, feed_dict=feed_dict)
    return distances

def _form_class_cluster(train_prs, num_class):
    co_act_tups = []
    co_targets = []
    co_predictions = []
    co_was_correct = []
    for i in xrange(num_class):
        inds_of_class_i = np.argwhere(train_prs.targets == i)[:, 0]
        co_act_tups.append(train_prs.pre_final[inds_of_class_i])
        co_targets.append(train_prs.targets[inds_of_class_i])
        co_predictions.append(train_prs.predictions[inds_of_class_i])
        co_was_correct.append(train_prs.was_correct[inds_of_class_i])
    return ClassOrganised(co_act_tups, co_targets, co_predictions ,co_was_correct)


def _visualise_all(kn_results, cc_results):
    for knr, cc_dists in zip(kn_results, cc_results):
        vis.show_neighbouring_instances_conv(knr, cc_dists)

class ClassOrganised:

    def __init__(self, co_act_tups, co_targets, co_predictions ,co_was_correct):
        self.act_tups = co_act_tups
        self.targets = co_targets
        self.predictions = co_predictions
        self.was_correct = co_was_correct




