import tensorflow as tf
import numpy as np


class TupleDistanceKNN:
    """Computation graph that computes nearest neighbour based on tuple distance"""

    def __init__(self, k, all_hidden):
        self.k = k
        self.all_regions = tf.placeholder(tf.bool, shape=[None, all_hidden], name="all_regions")
        self.predicted_region = tf.placeholder(tf.bool, shape=[all_hidden], name="predicted_region")

    def distances_and_indices(self):
        "Get the indices of the K nearest active regions to the current prediction."
        all_matches = tf.logical_xor(self.predicted_region, self.all_regions)
        #Cast to integers
        all_matches = tf.cast(all_matches, tf.int32, name="To_integers")
        distance_from_pred = tf.reduce_sum(all_matches, axis=1)
        distances, indices = tf.nn.top_k(tf.negative(distance_from_pred), k=self.k)
        return tf.negative(distances), indices


class SimpleNetworkDistanceKNN:
    """Computation graph that computes nearest neighbour based on the mean squared distance
    between activations"""

    def __init__(self, k, all_hidden):
        self.k = k
        self.all_regions = tf.placeholder(tf.float32, shape=[None, all_hidden], name="all_regions")
        self.all_targets = tf.placeholder(tf.int32, shape=[None], name="all_targets")
        self.all_predicted = tf.placeholder(tf.int32, shape=[None], name="all_predicted")
        self.predicted_region = tf.placeholder(tf.float32, shape=[all_hidden], name="predicted_region")
        self.predicted_class = tf.placeholder(tf.int32, shape=[], name="predicted_class")
        self.n = tf.placeholder(tf.int32, shape=[], name="n")

    def distances_and_indices(self, just_pred_and_correct=False):
        "Get the indices of the K nearest active regions to the current prediction."
        all_distances = tf.sqrt(tf.reduce_sum(tf.pow(self.all_regions - self.predicted_region, 2), axis=1))
        if just_pred_and_correct:
            all_distances += self._dist_hack()
        distances, indices = tf.nn.top_k(tf.negative(all_distances), k=self.k)
        return tf.negative(distances), indices

    def _dist_hack(self):
        is_pred = tf.equal(self.all_predicted, self.predicted_class)
        is_correct = tf.equal(self.all_targets, self.all_predicted)
        is_pred_and_correct = tf.logical_and(is_pred, is_correct)
        zeros = tf.zeros(shape=[self.n])
        infs = tf.multiply(float('inf'), tf.ones(shape=[self.n]))
        return tf.where(is_pred_and_correct, zeros, infs)

class KNearest:

    def __init__(self, conf):
        all_hidden = sum(conf.sizes[1:-1])
        self.KNN = _create_KNearest(conf.k, all_hidden, conf.is_binary)

    def report_simple_stats(self, sess, prediction, pred_class, final_reg_set):
        """Report the 4 simple statistics as specified in simple_logistic.py for a batch of predictions"""
        #TODO Chane to distances pc
        nearest_instances, nearest_regions, distances, _ = self.report_KNearest(sess, prediction, pred_class, final_reg_set)

        # 1) Compute distance:
        avg_dist = np.mean(distances)

        # 2) Count predicted and correct
        pred_and_correct = 0
        if pred_class in nearest_instances.keys():
            pred_and_correct =  nearest_instances[pred_class].correct

        # 3) and 4) - Compute inverse class entropy and number of incorrect predictions
        inv_pred_entropy = 0
        grand_total_inst = 0
        incorrect_neighbours = 0
        for predicted, inst_tracker in nearest_instances.iteritems():
            # 3)
            inst_count = inst_tracker.total_instances()
            inv_pred_entropy += inst_count ** 2.0
            grand_total_inst += inst_count
            # 4)
            incorrect_neighbours += inst_tracker.incorrect

        pred_and_correct /= float(grand_total_inst)
        inv_pred_entropy /= float(grand_total_inst ** 2.0)
        incorrect_neighbours /= float(grand_total_inst)

        return SimpleRegionStats(avg_dist, pred_and_correct, inv_pred_entropy, incorrect_neighbours)

    def report_KNearest(self, sess, prediction, pred_class, final_reg_set):
        """Returns a map of instances, corresponding to the K nearest active
        regions to a given prediction

        :param sess- The tf session

        :param prediction - The region for the prediction which we want to see
        the K Nearest neighbours of

        :param pred_class - The predicted class

        :param final_reg_set - All the final, active linear regions of
        the test set

        :return A map with keys that correspond to the class of each instance that was
        found in the K Nearest Activation regions. The value is a count of how many
        times that class occured in those regions.
        """
        final_regions = final_reg_set.final_regions
        Ts = final_reg_set.Ts
        all_targets = final_reg_set.all_targets
        all_predicted = final_reg_set.all_predicted
        n = len(final_regions)
        feed_dict = {self.KNN.predicted_region: prediction, self.KNN.all_regions: Ts,
                     self.KNN.all_targets: all_targets, self.KNN.all_predicted: all_predicted,
                     self.KNN.predicted_class: pred_class, self.KNN.n: n}

        #Get K-Nearest regions
        knn_all = self.KNN.distances_and_indices(False)
        distances, nearest_indicies = sess.run(knn_all, feed_dict=feed_dict)
        nearest_regions = [final_regions[i] for i in nearest_indicies]

        #Get K-NearestRegions that are predicted and correct
        knn_pc = self.KNN.distances_and_indices(True)
        distances_pc, nearest_indicies_pc = sess.run(knn_pc, feed_dict=feed_dict)
        avg_dist_pc = np.mean(distances_pc)
        #nearest_regions_pc = [final_regions[i] for i in nearest_indicies_pc]

        #Compose map of instances
        nearest_instances = {}
        #nearest_instances[pred_class] = InstanceTracker()
        for reg in nearest_regions:
            for pred in reg.predictions:
                target = pred.target
                predicted = pred.predicted
                if target not in nearest_instances.keys():
                    nearest_instances[target] = InstanceTracker()
                nearest_instances[target].increment(target == predicted)

        return nearest_instances, nearest_regions, distances, avg_dist_pc

class InstanceTracker:

    def __init__(self):
        self.correct = 0
        self. incorrect = 0

    def increment(self, is_correct):
        if is_correct:
            self.correct += 1
        else:
            self.incorrect += 1

    def total_instances(self):
        return self.correct + self.incorrect

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.correct != other.correct:
            return False
        if self.incorrect != other.incorrect:
            return False
        return True

    def __ne__(self, other):
        return self.__eq__(other)

    def __str__(self):
        return str(self.correct)+", "+str(self.incorrect)

class SimpleRegionStats:

    def __init__(self, avg_dist, pred_and_correct, inv_pred_entropy, incorrect_neighbours):
        self.avg_dist = avg_dist
        self.pred_and_correct = pred_and_correct
        self.inv_pred_entropy = inv_pred_entropy
        self.incorrect_neighbours = incorrect_neighbours

    def as_vector(self):
        return np.array([self.avg_dist,
                        self.pred_and_correct,
                        self.inv_pred_entropy,
                        self.incorrect_neighbours]).reshape(1, 4)


def _create_KNearest(k, all_hidden, is_binary):
    if is_binary:
        return TupleDistanceKNN(k, all_hidden)
    return SimpleNetworkDistanceKNN(k, all_hidden)


# replicated_regions = tf.reshape(self.all_regions, [-1, 1, self.all_hidden])
# replicated_regions = tf.tile(replicated_regions, [1, self.m, 1])
# all_matches = tf.logical_not(tf.logical_xor(self.predicted_region, replicated_regions))



