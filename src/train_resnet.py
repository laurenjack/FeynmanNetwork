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
tf.app.flags.DEFINE_boolean('minimal_summaries', False,
                            'produce fewer summaries to save HD space')


def top_k_error(predictions, labels, m, k=1):
    batch_size = float(m) #tf.shape(predictions)[0]
    is_correct = tf.equal(tf.cast(tf.argmax(predictions, 1), tf.int32), labels)
    #return tf.reduce_mean(tf.cast(is_correct, tf.float32)), self.z, self.a_out
    #in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k))
    in_top1 = tf.to_float(is_correct)
    num_correct = tf.reduce_sum(in_top1)
    return (batch_size - num_correct) / batch_size


def train(resnet, conf, resume):
    out_dir = conf.out_dir
    in_dir = conf.in_dir
    m = conf.m
    n = conf.n
    num_class = conf.num_classes
    lr = conf.lr
    lr_reduce_factor = conf.lr_reduce_factor
    lr_reduce_steps = conf.lr_reduce_steps
    lr_tensor = tf.placeholder(tf.float32, shape=[], name='learning_rate')
    #momentum = conf.momentum
    val_step = tf.get_variable('val_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    logits = resnet.inference()
    loss_ = resnet.loss
    predictions = tf.nn.softmax(logits)
    tau_op = resnet.W
    x_bar_op = resnet.x_bar

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
    opt = tf.train.AdamOptimizer(learning_rate=lr_tensor, beta1=0.9, beta2=0.999, epsilon=1e-30, name='Adam')
    opt_gen = tf.train.AdamOptimizer(learning_rate=lr_tensor, beta1=0.9, beta2=0.999, epsilon=1e-30)
    grads = opt.compute_gradients(loss_)
    for i in xrange(len(grads)):
        grad, var = grads[i]
        if grad is not None and not FLAGS.minimal_summaries:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    apply_gradient_op = opt.apply_gradients(grads, global_step=resnet.global_step)

    if not FLAGS.minimal_summaries:
        # Display the training images in the visualizer.
        tf.summary.image('images', resnet.images)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, batchnorm_updates_op)
    pre_z_and_z_op = resnet.pre_z_and_z()

    #x_bar_save_op = resnet.save_x_bar()
    saver = tf.train.Saver(tf.global_variables())

    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(out_dir, sess.graph)

    if resume:
        # Load network
        latest = tf.train.latest_checkpoint(in_dir)
        if not latest:
            print "No checkpoint to continue from in", in_dir
            sys.exit(1)
        print "resume", latest
        saver.restore(sess, latest)



    # actual_images = sess.run(images, {resnet.is_training: True, lr_tensor: lr})
    # c1 = actual_images[3, :, :, 0]
    # c2 = actual_images[3, :, :, 1]
    # c3 = actual_images[3, :, :, 2]

    # z_d = conf.stacks[-1].in_d
    # #x_bar_shape = (z_d, conf.num_classes)
    # x_bar = np.random.randn(z_d, conf.num_classes) / float(z_d)
    # x_bar = x_bar.astype(np.float32)

    for x in xrange(conf.max_steps):
        start_time = time.time()


        step = sess.run(resnet.global_step)
        i = [train_op, loss_, pre_z_and_z_op]

        write_summary = step % 100 and step > 1
        if write_summary:
            i.append(summary_op)

        if x in lr_reduce_steps:
            lr *= lr_reduce_factor

        #Retrieve tau and x_bar for outside network generation (if we did in network, it would train based
        #on the generation process, not what we want
        z_d = resnet.z_d
        tau, x_bar = sess.run([tau_op, x_bar_op])
        all_labels = np.array(reduce(lambda a, b: a+b, [[j] * 20 for j in xrange(10)])).astype(np.int32)
        indicies = np.arange(20*num_class)
        chosen_indicies = np.random.choice(indicies, size=m, replace=False)
        standard_normals = np.random.randn(20, z_d, num_class)
        tau = abs(tau.reshape(1, z_d, conf.num_classes))
        x_bar.reshape(1, z_d, conf.num_classes)
        gen_zs_classwise = x_bar + 1.0 / tau * standard_normals
        gen_zs_trans = gen_zs_classwise.transpose(0,2,1)
        gen_zs_flattened = gen_zs_trans.reshape(20*num_class, z_d)
        gen_zs = gen_zs_flattened[chosen_indicies]
        labels = all_labels[chosen_indicies]
        #Put gend zs in network and train accordingly
        o = sess.run(i, {resnet.is_training: True, resnet.Ind: 0.0, lr_tensor: lr, resnet.gen_zs: gen_zs, resnet.labels_for_gen: labels})
        # else:
        #     fake_zs = np.ones((128, 64)).astype(dtype=np.float32)
        #     fake_labels = np.ones(m).astype(dtype=np.int32)
        #     o = sess.run(i, {resnet.is_training: True, resnet.Ind: 0.0, lr_tensor: lr, resnet.gen_zs: fake_zs, resnet.labels_for_gen: fake_labels})

        #Increase lr by one order of mag for lone rbf training
        # if x == conf.max_steps:
        #     lr *= 10.0

        loss_value = o[1]
        x_diff, pre_z, pz, pa, pneg_dist, test_grad_tup = o[2]
        test_grads, test_grads_other = test_grad_tup
        # _, _, preds, ws, c1, c2 = o[2]
        #x_bar = update_x_bar(x_bar, labelss, zss, num_class, m)

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
            checkpoint_path = os.path.join(out_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=resnet.global_step)

        # Run validation periodically
        if step > 1 and step % 100 == 0:
            _, top1_error_value = sess.run([val_op, top1_error], {resnet.is_training: False })
            print('Validation top1 error %.2f' % top1_error_value)

def _use_generated_grads(global_step, max_steps, var):
    return tf.logical_and(global_step > max_steps, tf.logical_or('x_bar' in var.op.name, 'sd' in var.op.name))



    #sess.run(x_bar_save_op, feed_dict={resnet.x_bar: x_bar, resnet.is_training: True, lr_tensor: lr})
            

# def update_x_bar(x_bar, labels, zs, num_class, m):
#     """Note: This is a non tensorflow function"""
#     #First epoch, careful calculation of sample counts required
#     for t in xrange(num_class):
#         inds_of_t = np.argwhere(labels == t)[:, 0]
#         x_bar_for_t = x_bar[:, t]
#         z_for_t = zs[inds_of_t]
#         ss = z_for_t.shape[0]
#         update_ratio = ss/float(m)
#         update_x_bar_t = np.mean(z_for_t, axis=0)
#         new_x_bar_for_t = (1.0 - update_ratio) * x_bar_for_t + update_ratio * update_x_bar_t
#         x_bar[:, t] = new_x_bar_for_t
#     return x_bar


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






