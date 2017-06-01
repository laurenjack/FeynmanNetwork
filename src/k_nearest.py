import tensorflow as tf

class KNearest:

    def __init__(self, conf):
        all_hidden = sum(conf.sizes[1:-1])
        self.k = conf.k
        self.all_regions = tf.placeholder(tf.bool, shape=[None, all_hidden], name="all_regions")
        self.predicted_region = tf.placeholder(tf.bool, shape=[all_hidden], name="all_regions")

    def indices_of(self):
        "Get the indices of the K nearest active regions to the current prediction."
        all_matches = tf.logical_not(tf.logical_xor(self.predicted_region, self.all_regions))
        distance_from_pred = tf.negative(tf.reduce_sum(all_matches, axis=1))
        _, indices = tf.nn.top_k(distance_from_pred, k=self.k)
        return indices
