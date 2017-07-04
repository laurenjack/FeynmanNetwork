from tensorflow.examples.tutorials.mnist import input_data

class Config:

    def __init__(self, sizes, learning_rate, m, epochs, feyn_lr=0.01, feyn_epochs=20, k=10, epsilon=0.25, is_binary=True):
        self.sizes = sizes
        self.learning_rate = learning_rate
        self.m = m
        self.epochs = epochs
        self.feyn_lr = feyn_lr
        self.feyn_epochs = feyn_epochs
        self.k = k
        self.epsilon = epsilon
        self.is_binary = is_binary

        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        train_set = mnist.train
        val_set = mnist.validation
        self.X_val = val_set.images#[:2000]
        self.Y_val = val_set.labels#[:2000]
        self.X = train_set.images#[:2000]
        self.Y = train_set.labels#[:2000]
