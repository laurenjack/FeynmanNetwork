from train import train_and_report
from network import FeedForward
from conf import Config
from visualisation import *
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
train_set = mnist.train
X = train_set.images#[:2000]
Y = train_set.labels#[:2000]

sizes = [784, 30, 30, 10]
learning_rate = 0.1
m = 20
epochs = 5
conf = Config(sizes, learning_rate, m, epochs)
network = FeedForward(conf)

region_forest = train_and_report(network, X, Y, conf)
num_pred_map = region_forest.get_frequency_each_region(1)
frequency_of_instance_per_region(num_pred_map)