import tensorflow as tf
import numpy as np

class KNearest:

    def __init__(self, conf):
        all_hidden = sum(conf.sizes[1:-1])
        self.k = conf.k
        self.all_regions = tf.placeholder(tf.bool, shape=[None, all_hidden], name="all_regions")
        self.predicted_region = tf.placeholder(tf.bool, shape=[all_hidden], name="all_regions")

    def indices_of(self):
        "Get the indices of the K nearest active regions to the current prediction."
        all_matches = tf.logical_not(tf.logical_xor(self.predicted_region, self.all_regions))
        #Cast to integers
        all_matches = tf.cast(all_matches, tf.int32, name="To_integers")
        distance_from_pred = tf.reduce_sum(all_matches, axis=1)
        similarilties, indices = tf.nn.top_k(distance_from_pred, k=self.k)
        return similarilties, indices

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
        feed_dict = {self.predicted_region: prediction, self.all_regions: Ts}

        #Get K-Nearest regions
        similarilties, nearest_indicies = sess.run(self.indices_of(), feed_dict=feed_dict)
        nearest_regions = [final_regions[i] for i in nearest_indicies]

        #Compose map of instances
        nearest_instances = {}
        nearest_instances[pred_class] = InstanceTracker()
        for reg in nearest_regions:
            for pred in reg.predictions:
                target = pred.target
                predicted = pred.predicted
                if target not in nearest_instances.keys():
                    nearest_instances[target] = InstanceTracker()
                nearest_instances[target].increment(target == predicted)

        return nearest_instances, np.mean(similarilties)

class InstanceTracker:

    def __init__(self):
        self.correct = 0
        self. incorrect = 0

    def increment(self, is_correct):
        if is_correct:
            self.correct += 1
        else:
            self.incorrect += 1

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



