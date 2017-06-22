import tensorflow as tf
NUM_INPUTS = 4
NUM_OUTPUTS = 2


class SimpleLogistic:
    """A Feynman classifier used to predict the probability of being correct using a logistic
    regression.

    This classifier is simple in the sense that it has 4 inputs, all region statistics
    from it K-Nearest neighbours:

    1) Distance from prediction - The average distance the neighbours have from the current
    prediction.

    2) Predicted and Correct Count - The number of neighbours that were predicted and
    are correct

    3) Class entropy - The weighted average number of examples per class within the
    K - Nearest neighbours

    4) Incorrect neighbours - The total """

    def __init__ (self, conf):
        self.learning_rate = conf.feyn_lr
        #Specify input vector
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, NUM_INPUTS], name="inputs")
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, NUM_OUTPUTS], name="target")
        #Define the logistic regression model
        self.W = tf.get_variable("W_simple_logitstic", shape=[NUM_INPUTS, NUM_OUTPUTS], initializer=tf.contrib.layers.xavier_initializer())
        self.b = tf.Variable(0.1, dtype=tf.float32, name="b_simple_logistic")
        z = tf.add(tf.matmul(self.x, self.W), self.b)
        self.a = tf.nn.sigmoid(z)
        #Cost function for training
        self.cost = tf.reduce_mean(-self.y * tf.log(self.a))# + -(1.0 - self.y) * tf.log(1.0 - self.a))

    def train(self):
        return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

    def wb(self):
        return self.W, self.b

    def accuracy(self):
        is_correct = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.a, 1))
        return tf.reduce_mean(tf.cast(is_correct, tf.float32))
        # is_correct = tf.equal(self.y, tf.round(self.a))
        # return tf.reduce_mean(tf.cast(is_correct, tf.float32))
