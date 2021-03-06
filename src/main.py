from train import *
from network import FeedForward
from conf import Config
import feynman_classifier
import human_classifer
from visualisation import *
from tensorflow.examples.tutorials.mnist import input_data
from k_nearest import KNearest

#tf.set_random_seed(653712155)
#np.random.seed(65331165)
class Counter:

    def __init__(self):
        self.count = 0

    def inc(self):
        self.count += 1

NET_GLOBAL = Counter()

sizes = [784, 64, 10]
learning_rate = 0.001
m = 20
epochs = 20
conf = Config(sizes, learning_rate, m, epochs, feyn_lr=0.05, feyn_epochs=100, k=15, epsilon=0.05, is_binary=True,
              is_w_pixels=False, is_rbf=True, log_dir='/home/laurenjack/mnist_logs')#/255.0)
network = FeedForward(conf, NET_GLOBAL)
X = conf.X
Y = conf.Y

#Train the network and retrieyve the info regarding its active linear regions
train_region_forest, sess = train_and_report(network, X, Y, conf)
train_region_set = train_region_forest.all_final_regions()

human_classifer.run(network, train_region_set, conf, sess, test_samples=30)

#Get the info regarding the active linear regions of the validation setf
# X_val = conf.X_val
# Y_val = conf.Y_val
# val_region_forest = report(network, X_val, Y_val, conf, sess)
#sess.close()
#feynman_classifier.run(train_region_forest, val_region_forest, conf, feyn_n=500)



#num_pred_map = region_forest.get_frequency_each_region(1)
#frequency_of_instance_per_region(num_pred_map)