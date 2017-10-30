import tensorflow as tf


class DistanceComputer:
    """Computes the distance of a given test point, to a batch of training examples"""

    def __init__(self):
        self.train_points = tf.placeholder(tf.float32, shape=[None, 1136], name="train_points")
        self.test_point = tf.placeholder(tf.float32, shape=[1136], name="test_point")
        #self.ftws = tf.placeholder(tf.float32, shape=[None, 1136], name="train_target_weights")

    def distances(self):
        p = tf.reduce_mean(self.train_points, axis=0)
        var = tf.reduce_mean(tf.pow(self.train_points - p, 2.0), axis=0)
        weights = 1.0 / (var + 10 ** -2) #(p + 10 ** -1) * (1.0 - p + 10 ** -1)
        diff = weights * (self.train_points - self.test_point)
        return tf.reduce_sum(tf.abs(diff), axis=1)
        #return tf.sqrt(tf.reduce_sum(tf.pow(diff, 2), axis=1))
        # diffs = tf.logical_xor(self.test_point, self.train_points)
        # return tf.count_nonzero(diffs, axis=1, dtype=tf.int32)



class KNearest:


    def __init__(self):
        self.distances = tf.placeholder(tf.float32, shape=[None], name="distances")
        self.their_targets = tf.placeholder(tf.int32, shape=[None], name="their_targets")
        self.their_predictions = tf.placeholder(tf.int32, shape=[None], name="their_predictions")
        self.train_was_correct = tf.placeholder(tf.bool, shape=[None], name="train_was_correct")

    def find_k_nearest(self, k):
        #Compute distances
        #distances = tf.sqrt(tf.reduce_sum(tf.pow(self.train_points - self.test_point, 2), axis=1))
        #all_diffs = tf.cast(all_diffs, tf.int32, name="To_integers")
        #distances = tf.reduce_sum(all_diffs, axis=1)
        #Find the nearest neighbours
        their_distances, indices = tf.nn.top_k(tf.negative(self.distances), k=k)
        their_distances = tf.negative(their_distances)
        #Get the relevant info relating to the nearest neighbours
        nearest_targets = tf.gather(self.their_targets, indices)
        nearest_predictions = tf.gather(self.their_predictions, indices)
        were_they_correct = tf.gather(self.train_was_correct, indices)
        avg_distance = tf.reduce_mean(their_distances)

        return nearest_targets, nearest_predictions, their_distances, were_they_correct, avg_distance, indices

class TrainCounter:

    def __init__(self, k, all_hidden, num_classes):
        self.k = k
        self.class_clusters = [tf.placeholder(tf.float32, shape=[None, all_hidden], name="cluster"+str(i)) for i in xrange(num_classes)]
        self.test_point = tf.placeholder(tf.float32, shape=[all_hidden], name="tc_test_point")

    def find_count_dist(self):
        man_hat_dists = []
        for cc in self.class_clusters:
            p = tf.reduce_mean(cc, axis=0)
            var = tf.reduce_mean(tf.pow(cc - p, 2.0), axis=0)
            weights = 1.0 / (var + 10 ** -2) #(p + 10 ** -1) * (1.0 - p + 10 ** -1)
            test_point_as_float = tf.cast(self.test_point, tf.float32)
            man_hat_dist = tf.reduce_sum(tf.abs(weights*(p - test_point_as_float)))
            man_hat_dists.append(man_hat_dist)
        return man_hat_dists


class KNearestInfo:

    def __init__(self, nearest_targets, nearest_predictions, their_distances, were_they_correct, avg_distance):
        self.nearest_targets = nearest_targets
        self.nearest_predictions = nearest_predictions
        self.their_distances = their_distances
        self.were_they_correct = were_they_correct
        self.avg_distance = avg_distance
        self.target = None
        self.prediction = None
