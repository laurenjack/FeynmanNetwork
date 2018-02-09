from conf import *
from resnet import *
from train_resnet import *
from prediction_probs import *
from intersection_diagnostics import *
import cifar
import resnet_human_classifier as hc

conf = ConfBuilder()

is_network_train = False
conf.is_rbf_soft = True

np.set_printoptions(precision=2, suppress=True)

#tf.set_random_seed(653712155)
#np.random.seed(65331165)

#Data_params
conf.num_classes = 10
conf.image_dims = 32
conf.train_dir = '/home/laurenjack/models'
conf.n = 50000

#Training parameters
conf.lr = 0.001
conf.lr_reduce_factor = 0.1
conf.lr_reduce_steps = [32000, 48000]
conf.momentum = 0.9
conf.m = 128
#Report m (batch_size) used for when we want to report on the entire data set (usually invlolves)
#iterating over the entire thing once. Hence it is simply a number configured to conform to the size
#of the training set
conf.reporting_m = 200
# conf.epochs = 100
conf.max_steps = 64000

#Network Structural parameters
conf.num_filter_list = [16, 16, 32, 64]
conf.num_block_list = [5, 5, 5]
conf.pp_k_size = 3
conf.pp_stride = 1
conf.has_mp = False
conf.k_size = 3
conf.f_stride = 2
conf.stride = 1
conf.adv_epsilon = 0.01

#Accuary prediction params
conf.k = 10
conf.s = 20

conf = conf.build()

is_training = tf.placeholder('bool', [], name='is_training')
global_step = tf.get_variable('global_step', [],
                              initializer=tf.constant_initializer(0),
                              trainable=False)

if is_network_train:
    #Load the data
    cifar10 = cifar.Cifar10()
    images_train, labels_train = cifar10.distorted_inputs(conf.m)
    images_val, labels_val = cifar10.inputs(True, conf.m)

    images, labels = tf.cond(is_training,
            lambda: (images_train, labels_train),
            lambda: (images_val, labels_val))

    #Build the network
    resnet = Resnet(conf, is_training, global_step, images, labels)

    #Train the network
    x_bar = train(resnet, conf)
else:
    #x_bar_vs_centre_resnet(conf, is_training, global_step)
    report_prediction_probs(conf, is_training, global_step, x_bar)
    #Load the latest network, and perform k_nearest inference
    #hc.run(conf, is_training, global_step)


