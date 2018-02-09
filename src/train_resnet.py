import tensorflow as tf
import numpy as np
from resnet import *

import datetime
import numpy as np
import os
import time
import sys

FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string('train_dir', '/tmp/resnet_train',
#                            """Directory where to write event logs """
#                            """and checkpoint.""")
tf.app.flags.DEFINE_boolean('resume', False,
                            'resume from latest saved state')
tf.app.flags.DEFINE_boolean('minimal_summaries', True,
                            'produce fewer summaries to save HD space')


def top_k_error(predictions, labels, m, k=1):
    batch_size = float(m) #tf.shape(predictions)[0]
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k))
    num_correct = tf.reduce_sum(in_top1)
    return (batch_size - num_correct) / batch_size


def train(resnet, conf):
    train_dir = conf.train_dir
    m = conf.m
    lr = conf.lr
    lr_reduce_factor = conf.lr_reduce_factor
    lr_reduce_steps = conf.lr_reduce_steps
    lr_tensor = tf.placeholder(tf.float32, shape=[], name='learning_rate')
    momentum = conf.momentum
    val_step = tf.get_variable('val_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    logits = resnet.inference()
    loss_ = resnet.loss
    predictions = tf.nn.softmax(logits)

    top1_error = top_k_error(predictions, resnet.labels, m, 1)


    # loss_avg
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, resnet.global_step)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([loss_]))
    tf.summary.scalar('loss_avg', ema.average(loss_))

    # validation stats
    ema = tf.train.ExponentialMovingAverage(0.9, val_step)
    val_op = tf.group(val_step.assign_add(1), ema.apply([top1_error]))
    top1_error_avg = ema.average(top1_error)
    tf.summary.scalar('val_top1_error_avg', top1_error_avg)

    #tf.summary.scalar('learning_rate', lr)

    #opt = tf.train.MomentumOptimizer(lr_tensor, momentum)
    opt = tf.train.AdamOptimizer(learning_rate=lr_tensor, beta1=0.9, beta2=0.999, epsilon=1e-08)
    grads = opt.compute_gradients(loss_)
    for grad, var in grads:
        if grad is not None and not FLAGS.minimal_summaries:
            tf.histogram_summary(var.op.name + '/gradients', grad)
    apply_gradient_op = opt.apply_gradients(grads, global_step=resnet.global_step)

    if not FLAGS.minimal_summaries:
        # Display the training images in the visualizer.
        tf.image_summary('images', resnet.images)

        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)

    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

    saver = tf.train.Saver(tf.global_variables())

    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

    if FLAGS.resume:
        latest = tf.train.latest_checkpoint(train_dir)
        if not latest:
            print "No checkpoint to continue from in", train_dir
            sys.exit(1)
        print "resume", latest
        saver.restore(sess, latest)

    # actual_images = sess.run(images, {resnet.is_training: True, lr_tensor: lr})
    # c1 = actual_images[3, :, :, 0]
    # c2 = actual_images[3, :, :, 1]
    # c3 = actual_images[3, :, :, 2]

    z_d = conf.stacks[-1].in_d
    #x_bar_shape = (z_d, conf.num_classes)
    x_bar = np.random.randn(z_d, conf.num_classes) / float(z_d)
    x_bar = x_bar.astype(np.float32)
    per_class_sample_counts = np.zeros(conf.num_classes).astype(np.float32)


    for x in xrange(conf.max_steps + 1):
        start_time = time.time()

        step = sess.run(resnet.global_step)
        i = [train_op, loss_, resnet.z_and_labels()]

        write_summary = step % 100 and step > 1
        if write_summary:
            i.append(summary_op)

        if x in lr_reduce_steps:
            lr *= lr_reduce_factor

        o = sess.run(i, {resnet.x_bar: x_bar, resnet.is_training: True, lr_tensor: lr})



        loss_value = o[1]

        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 5 == 0:
            examples_per_sec = float(conf.m) / float(duration)
            format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (step, loss_value, examples_per_sec, duration))

        if write_summary:
            summary_str = o[3]
            summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step > 1 and step % 100 == 0:
            checkpoint_path = os.path.join(train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=resnet.global_step)

        # Run validation periodically
        if step > 1 and step % 100 == 0:
            _, top1_error_value = sess.run([val_op, top1_error], {resnet.x_bar: x_bar, resnet.is_training: False })
            print('Validation top1 error %.2f' % top1_error_value)
            

def update_x_bar(x_bar, labels, zs, num_class, e, per_class_sample_counts):
    """Note: This is a non tensorflow function"""
    #First epoch, careful calculation of sample counts required
    if e == 0:
        for t in xrange(num_class):
            inds_of_t = np.argwhere(labels == t)[:, 0]
            x_bar_for_t = x_bar[:, t]
            z_for_t = zs[inds_of_t]
    else:
        for t in xrange(num_class):
            inds_of_t = np.argwhere(labels == t)[:, 0]
            x_bar_for_t = x_bar[:, t]
            z_for_t = zs[inds_of_t]


   # #Adam params
   #  b1 = 0.9
   #  b2 = 0.999
   #  ep = 10.0 ** -20
   #  #Adam variables
   #  fan_in = conf.stacks[-1].in_d
   #  x_bar_shape = (fan_in, conf.num_classes)
   #  x_bar = np.random.randn(fan_in, conf.num_classes) / float(fan_in)
   #  x_bar = x_bar.astype(np.float32)
   #  momentum = np.zeros(x_bar_shape)
   #  v = np.zeros(x_bar_shape)

    # # Perform adam update of x_bar
    # zs, np_labels = o[2]
    # dC_dx_bar = resnet.d_centre_dx_bar(x_bar, np_labels, zs, conf.num_classes)
    # t = step + 1
    # momentum = b1 * momentum + (1 - b1) * dC_dx_bar
    # v = b2 * v + (1 - b2) * dC_dx_bar ** 2.0
    # x_bar -= lr * (1 - b2 ** t) ** 0.5 / (1 - b1 ** t) * momentum / (v ** 0.5 + ep)






