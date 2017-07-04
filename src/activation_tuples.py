import numpy as np
from linear_regions import Prediction

"""This file contains the real value analgoues to the various classes in linear_regions.py

This file holds the classes which maintain the real valued activation tuples for each
instance of a data set. As a consequence the notion of each being a region is erroneous,
as each tuple in this case represents a point. However, for the sake of code re-use,
we re-implement the region based interface here, at the cost of a sensibly named interface."""

class TupleSetBuilder:
    """Real valued analogue of RegionSetBuilder"""

    def __init__(self, conf):
        self.activation_tuples = []

    def putRegion(self, A_by_layer, targets, predictions):
        all_A = self._transpose(A_by_layer)
        for i in xrange(len(all_A)):
            target = targets[i]
            predicted = predictions[i]
            prediction = Prediction(target, predicted)
            acts = np.concatenate(all_A[i])
            act_tup = ActivationTuple(prediction, acts)
            self.activation_tuples.append(act_tup)

    def get_forest(self):
        real_tuple_set = RealTupleSet(self.activation_tuples)
        return RealTupleForest(real_tuple_set)

    def _transpose(self, T_by_layer):
        """Transpose the list T_by_layer to have a top level ordering based on each
         instance rather than each layer,

        :param T_by_layer - a list of matricies with dimensions num_layer x m x d_l
        (where m is the mini-batch size and d_l is the width of the lth layer)

        :return all_T  - a list of list of vectors with dimensions m x num_layer x d_l"""
        all_T = []
        n = len(T_by_layer)
        m = T_by_layer[0].shape[0]
        # Iterate over m mini_batches
        for b in xrange(m):
            T = []
            # Then over n layers
            for l in xrange(n):
                t = T_by_layer[l][b]
                T.append(t)
            all_T.append(T)
        return all_T

class RealTupleForest:
    """Data structure representing all the active regions in a network for
    a given test set"""

    def __init__(self, real_tuple_set):
        self.real_tuple_set = real_tuple_set

    def all_final_regions(self):
        return self.real_tuple_set

    def get_n_correct_and_incorrect(self, n):
        corr = []
        incorr = []
        for act_tuple in self.real_tuple_set.final_regions:
            if act_tuple.is_correct() and len(corr) < n:
                corr.append(act_tuple)
            if not act_tuple.is_correct() and len(incorr) < n:
                incorr.append(act_tuple)
        return RealTupleSet(corr), RealTupleSet(incorr)


class RealTupleSet:
    """Real-valued analogue of FinalRegionSet."""

    def __init__(self, activation_tuples):
        self.Ts = [at.acts for at in activation_tuples]
        self.final_regions = activation_tuples
        n = len(activation_tuples)
        self.all_targets = np.zeros(n)
        self.all_predicted = np.zeros(n)
        for i in xrange(n):
            pred = activation_tuples[i].predictions[0]
            self.all_targets[i] = pred.target
            self.all_predicted[i] = pred.predicted


class ActivationTuple:
    """Real valued analouge of Region/FinalRegion"""
    def __init__(self, prediction, acts):
        self.acts = acts
        self.predictions = [prediction]

    def is_correct(self):
        return self.predictions[0].is_correct()


class FreqTracker:
    def __init__(self):
        self.freq_perfect = 0
        self.freq_mixed = 0