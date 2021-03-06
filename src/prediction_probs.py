import tensorflow as tf
import numpy as np
import visualisation as vis
import sys
import cifar
from resnet import Resnet
from write_to_file import *
from scipy.misc import imsave, imshow


def report_prediction_probs(conf, is_training, global_step):
    train_dir = conf.out_dir
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
    #adv_op_z = resnet.fgsm_adv_example_z_space()
    pp_op = resnet.prediction_probs()
    #pp_op_gen = resnet.prediction_probs_gen()
    #pr_op =resnet.prediction_result()
    #labels_and_pre_z_op = resnet.labels_and_pre_z()
    #load_x_bar_op = resnet.x_bar_save
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

    #Load the stored x_bar
    #x_bar = sess.run(load_x_bar_op, feed_dict={is_training: False})

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

    # Save pre z observations of training set
    # m = 128
    # total_low = 0
    # inds = np.arange(m)
    # train_dict = {resnet.is_training: False}
    # all_data = []
    # for i in xrange(0, n, m):
    #     train_dict[images] = x[i:i+m]
    #     train_dict[targets] = y[i:i+m]
    #     labels, pre_z = sess.run(labels_and_pre_z_op, feed_dict=train_dict)
    #     labels = labels.astype(np.float32)
    #     batch_data = np.concatenate([labels, pre_z], axis=1)
    #     all_data.append(batch_data)
    #     # if i+m > n:
    #     #     inds = np.arange(n - i)
    #     # max_probs = pred_probs[inds, preds]
    #     # is_low = np.where(max_probs <= 0.5, 1, 0)
    #     # total_low += np.sum(is_low)
    # all_data_np = np.concatenate(all_data, axis=0)
    # csv_writer = CsvWriter()
    # csv_writer.write_to_csv(all_data_np)

    # Create adverserial examples based on the validation set
    x_adv = x_test[advs_inds]
    y_adv = y_test[advs_inds]
    adv_dict = {resnet.is_training: False, images: x_adv, targets: y_adv}
    x_adv = sess.run(adv_op, feed_dict=adv_dict)
    #z_adv = sess.run(adv_op_z, feed_dict=adv_dict)
    #Replace with perturbed images
    adv_dict[images] = x_adv
    #adv_dict[resnet.gen_zs] = z_adv

    #Report predictions of the val set
    val_prediction_probs, val_rbf_logits = sess.run(pp_op, feed_dict=val_dict)
    print "Val prediction probs:"
    print _report(y_test_samp, val_prediction_probs, val_rbf_logits, x_test_samp)
    print ""

    # Report predictions of the adverserial set
    adv_prediction_probs, adv_rbf_logits = sess.run(pp_op, feed_dict=adv_dict)
    #adv_prediction_probs = sess.run(pp_op_gen, feed_dict=adv_dict)
    print "Adverserial prediction probs:"
    print _report(y_adv, adv_prediction_probs, adv_rbf_logits, x_adv)

def _report(targets, predictions, rbf_logits, images):
    #targets = np.argmax(targets, axis=1)
    s = targets.shape[0]
    for i in xrange(s):
        print "Target: "+str(targets[i])+"    Prediction: "+str(predictions[i])
        print "RBF Logits: "+str(rbf_logits[i])
        print ""