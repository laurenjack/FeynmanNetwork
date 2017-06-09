from train import *
from network import FeedForward
from conf import Config
from visualisation import *
from tensorflow.examples.tutorials.mnist import input_data
from k_nearest import KNearest

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
train_set = mnist.train
val_set = mnist.validation
X_val = val_set.images #[:2000]
Y_val = val_set.labels #[:2000]
X = train_set.images#[:2000]
Y = train_set.labels#[:2000]

sizes = [784, 30, 30, 10]
learning_rate = 0.01
m = 20
epochs = 20
conf = Config(sizes, learning_rate, m, epochs, k=10)
network = FeedForward(conf)
test_samples = 20

#Train the network and retrieve the info regarding its active linear regions
test_region_forest, sess = train_and_report(network, X, Y, conf)
all_test = test_region_forest.all_final_regions()

#Find the regions for the validation set
val_region_forest = report(network, X_val, Y_val, conf, sess)

#Find the regions for random noise
X_noise = np.random.randn(test_samples, 784)
Y_noise = np.zeros((test_samples, 10))
garbage_forest = report(network, X_noise, Y_noise, conf, sess)
garbage_regions = garbage_forest.all_final_regions()


#Find equal part correctly classified and incorrectly classified examples from the training set
corr, incorr = val_region_forest.get_n_correct_and_incorrect(test_samples)
k_nearest = KNearest(conf)

def K_nearest_for_all(final_reg_set):
    final_regions = final_reg_set.final_regions
    Ts = final_reg_set.Ts
    for i in xrange(len(final_regions)):
        first_pred = final_regions[i].predictions[0]
        T = Ts[i]
        nearest, avg_sim = k_nearest.report_KNearest(sess, T, first_pred.predicted, all_test)
        show_neighbouring_instances(nearest,first_pred.predicted, first_pred.target, avg_sim)

K_nearest_for_all(garbage_regions)
K_nearest_for_all(corr)
K_nearest_for_all(incorr)


#num_pred_map = region_forest.get_frequency_each_region(1)
#frequency_of_instance_per_region(num_pred_map)