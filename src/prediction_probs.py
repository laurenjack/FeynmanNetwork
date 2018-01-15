import tensorflow as tf
import numpy as np
import visualisation as vis
import sys
import cifar
from resnet import Resnet

def report_prediction_probs(conf, is_training, global_step):
    train_dir = conf.train_dir
    im_dim = conf.image_dims
    num_classes = conf.num_classes
    k = conf.k
    s = conf.s

    # Create ops
    images = tf.placeholder(tf.float32, shape=[None, im_dim, im_dim, 3], name="image_input")
    targets = tf.placeholder(tf.int32, shape=[None], name="targets")
    resnet = Resnet(conf, is_training, global_step, images, targets)
    #rep = Reporter(conf, resnet)
    adv_op = resnet.fgsm_adverserial_example()
    pp_op = resnet.prediction_probs()
    # dc = DistanceComputer()
    # dc_op = dc.distances()
    # k_nearest = KNearest()
    # kn_op = k_nearest.find_k_nearest(k)
    # tc = TrainCounter(k, 1136, num_classes)
    # tc_op = tc.find_count_dist()

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

    # Sample part of validation set to gen adverserial examples and sample part for correct examples
    x, y = train
    x_test, y_test = test
    n = x.shape[0]
    n_test = x_test.shape[0]
    rand_inds = np.random.choice(np.arange(n_test), size=s * 2, replace=False)
    advs_inds = rand_inds[0:s]
    val_inds = rand_inds[s:]
    x_test_samp = x_test[val_inds]
    y_test_samp = y_test[val_inds]
    val_dict = {resnet.is_training: False, images: x_test_samp, targets: y_test_samp}

    # Create adverserial examples based on the validation set
    x_adv = x_test[advs_inds]
    y_adv = y_test[advs_inds]
    adv_dict = {resnet.is_training: False, images: x_adv, targets: y_adv}
    x_adv = sess.run(adv_op, feed_dict=adv_dict)
    #Replace with perturbed images
    adv_dict[images] = x_adv

    #Report predictions of the val set
    val_prediction_probs = sess.run(pp_op, feed_dict=val_dict)
    print "Val prediction probs:"
    print val_prediction_probs
    print ""

    # Report predictions of the adverserial set
    adv_prediction_probs = sess.run(pp_op, feed_dict=adv_dict)
    print "Adverserial prediction probs:"
    print adv_prediction_probs

    # # Report on the training set
    # train_prs, _ = rep.report_full_ds(x, y, sess, n)
    # print "Training set reported on"
    #
    # # Report on s test set instances
    # x_test_samp = x_test[val_inds]
    # y_test_samp = y_test[val_inds]
    # test_prs, _ = rep.report_full_ds(x_test_samp, y_test_samp, sess, s)
    #
    # # Report on the s adverserial examples
    # adv_prs, _ = rep.report_full_ds(x_adv, y_adv, sess, s)
    # print "Validation and adverserial samples reported on"

    # Create sets by class on training set
    class_organised = _form_class_cluster(train_prs, num_classes)

    # K nearest
    knrs_validation, tc_val = _k_nearest_for_each(k_nearest, kn_op, dc, dc_op, sess, train_prs, test_prs, tc, tc_op,
                                                  class_organised)  # ftws)
    knrs_adverserial, tc_adv = _k_nearest_for_each(k_nearest, kn_op, dc, dc_op, sess, train_prs, adv_prs, tc, tc_op,
                                                   class_organised)  # ftws)
    print "K Nearest complete"

    # Pass off to visualisation component
    print "Visualising validation samples"
    _visualise_all(knrs_validation, tc_val)
    print "Visualising adverseries"
    _visualise_all(knrs_adverserial, tc_adv)