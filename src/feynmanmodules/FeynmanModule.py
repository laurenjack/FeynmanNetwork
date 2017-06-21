import tensorflow as tf

class FeynmanModule:
    """An abstract class, every FeymanModule must predict the probability its network
    is correct, given the region statistics for a particular prediction"""

    def __init__(self, K):
        self.K = K
        self.inputs = tf.placeholder(tf.float32, shape=[None, 3])