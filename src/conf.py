class Config:

    def __init__(self, sizes, learning_rate, m, epochs, k=10, epsilon=0.25/255.0):
        self.sizes = sizes
        self.learning_rate = learning_rate
        self.m = m
        self.epochs = epochs
        self.k = k
        self.epsilon = epsilon
