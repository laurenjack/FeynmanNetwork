from train import *
from network import FeedForward
from conf import Config
import feynman_classifier
import human_classifer
from visualisation import *
from tensorflow.examples.tutorials.mnist import input_data
from k_nearest import KNearest


sizes = [784, 30, 30, 10]
learning_rate = 0.01
m = 20
epochs = 20
conf = Config(sizes, learning_rate, m, epochs, feyn_lr=0.05, feyn_epochs=100, k=10, epsilon=0.25, is_binary=False)#/255.0)
network = FeedForward(conf)
X = conf.X
Y = conf.Y

#Train the network and retrieve the info regarding its active linear regions
train_region_forest, sess = train_and_report(network, X, Y, conf)
train_region_set = train_region_forest.all_final_regions()

human_classifer.run(network, train_region_set, conf, sess, test_samples=10)

#Get the info regarding the active linear regions of the validation set
# X_val = conf.X_val
# Y_val = conf.Y_val
# val_region_forest = report(network, X_val, Y_val, conf, sess)
#sess.close()
#feynman_classifier.run(train_region_forest, val_region_forest, conf, feyn_n=500)



#num_pred_map = region_forest.get_frequency_each_region(1)
#frequency_of_instance_per_region(num_pred_map)