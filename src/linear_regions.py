import numpy as np

class RegionSetBuilder:

    def __init__(self, conf):
        self.sizes = conf.sizes
        self.num_layers = len(self.sizes) - 2
        self.hash_powers = self._create_hash_powers(self.sizes)
        self.roots = {}

    def putRegion(self, T_by_layer, targets, predictions):
        #Re-organise ts according to instances rather than layers
        all_T = self._transpose(T_by_layer)

        #Add each linear region to the Tree
        for i in xrange(len(all_T)):
            T = all_T[i]
            target = targets[i]
            predicted = predictions[i]
            l = 0
            is_present = True
            parent = None
            while l < self.num_layers and is_present:
                t = T[l]
                is_present, parent = self._is_present(t, l, parent)
                l += 1
            final_reg = self._fill_unique(l, T, parent)
            self._add_prediction_to(final_reg, target, predicted)

    def get_forest(self):
        return RegionForest(self.roots)

    def _is_present(self, t, l, parent):
        """Check if the current region t at layer l is already present in the region tree. If
        it is return true and the equivalent region. If it isn't return false and the newly
        created Region"""
        if parent is None:
            children = self.roots
        else:
            children = parent.sub_regions
        key = self._hash_t(t, l)
        if key in children.keys():
            return True, children[key]
        return False, self._new_region(l, t, parent, key)

    def _fill_unique(self, l0, T, parent):
        for l in xrange(l0, self.num_layers):
            t = T[l]
            key = self._hash_t(t, l)
            parent = self._new_region(l, t, parent, key)
        return parent

    def _new_region(self, l, t, parent, key):
        #Create the new region
        if l == self.num_layers - 1:
            new_region = FinalRegion(l, t, parent)
        else:
            new_region = Region(l, t, parent)
        #Add it to its parent
        if parent is None:
            self.roots[key] = new_region
        else:
            parent.sub_regions[key] = new_region
        return new_region

    def _add_prediction_to(self, final_reg, target, predicted):
        predictions = final_reg.predictions
        predictions.append(Prediction(target, predicted))

    def _transpose(self, T_by_layer):
        """Transpose the list T_by_layer to have a top level ordering based on each
         instance rather than each layer,
        
        :param T_by_layer - a list of matricies with dimensions num_layer x m x d_l
        (where m is the mini-batch size and d_l is the width of the lth layer)
        
        :return all_T  - a list of list of vectors with dimensions m x num_layer x d_l"""
        all_T = []
        n = len(T_by_layer)
        m = T_by_layer[0].shape[0]
        #Iterate over m mini_batches
        for b in xrange(m):
            T = []
            #Then over n layers
            for l in xrange(n):
                t = T_by_layer[l][b]
                T.append(t)
            all_T.append(T)
        return all_T

    def _create_hash_powers(self, sizes):
        """Create the powers that each index i of t is raised to, so that each t may be represented
        by a unique integer"""
        hash_powers = []
        for l in sizes[1:-1]:
            hp = np.zeros(shape=l, dtype=np.long)
            for i in xrange(l):
                hp[i] = 1 << (l - i - 1)
            hash_powers.append(hp)
        return hash_powers

    def _hash_t(self, t, l):
        return np.sum(self.hash_powers[l] * t)

class RegionForest:
    """Data structure representing all the active regions in a network for
    a given test set"""

    def __init__(self, roots):
        self.roots = roots

    def all_final_regions(self):
        final_regions = []
        self._all_final_regions(self.roots, final_regions)
        return FinalRegionSet(final_regions)

    def get_n_correct_and_incorrect(self, n):
        corr = []
        incorr = []
        self._get_n_correct_and_incorrect(self.roots, n, corr, incorr)
        return FinalRegionSet(corr), FinalRegionSet(incorr)

    def get_frequency_each_region(self, at_layer):
        num_pred_map = {}
        self._get_frequency_each_region(self.roots, num_pred_map, 1, at_layer)
        num_pred_map[1] = FreqTracker()
        return num_pred_map

    def _all_final_regions(self, regions, final_regions):
        for key, T in regions.iteritems():
            if isinstance(T, FinalRegion):
                final_regions.append(T)
            else:
                self._all_final_regions(T.sub_regions, final_regions)

    def _get_n_correct_and_incorrect(self, regions, n, corr, incorr):
        if len(corr) >= n and len(incorr) >=n:
            return
        for key, T in regions.iteritems():
            if isinstance(T, FinalRegion):
                if len(corr) < n and T.has_correct_pred():
                    corr.append(T)
                if len(incorr) < n and T.has_inc_pred():
                    incorr.append(T)
            else:
                self._get_n_correct_and_incorrect(T.sub_regions, n, corr, incorr)



    def _get_frequency_each_region(self, regions, num_pred_map, l, at_layer):
        for _, T in regions.iteritems():
            if l == at_layer:
                self._update_freq(T, num_pred_map)
            else:
                self._get_frequency_each_region(T.sub_regions, num_pred_map, l+1, at_layer)


    def _update_freq(self, T, num_pred_map):
        cor, inc = T.num_predictions()
        num_pred = cor + inc
        is_pure, _ = T.is_pure()
        if num_pred in num_pred_map.keys():
            freq_tracker = num_pred_map[num_pred]
        else:
            freq_tracker = FreqTracker()
            num_pred_map[num_pred] = freq_tracker
        if is_pure:
            freq_tracker.freq_perfect += 1
        else:
            freq_tracker.freq_mixed += 1




class FinalRegionSet:
    """Class representing all the final regions for a given RegionForest.
    
    It's properties are a list of all the final regions, as well as a numpy
    array with corresponding indicies, which contains the tuple set T for
    each region. For each region, this tuple set T is concatenated in a
    single numpy array."""

    def __init__(self, final_regions, Ts=None):
        self.final_regions = final_regions
        if Ts is None:
            Ts = map(lambda reg: reg.concatenated_T(), final_regions)
            self.Ts = np.array(Ts)
        else:
            self.Ts = Ts

class Region:
    """Data structure representing a linear region at a given layer"""
    def __init__(self, l, t, super_region):
        self.l = l
        self.t = t
        self.super_region = super_region
        self.sub_regions = {}

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        # if self.layer is None or self.id is None:
        #     raise TypeError("")
        if self.l != other.l:
            return False
        if not np.array_equal(self. t, other.t):
            return False
        if self.super_region != other.super_region:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        ts = []
        reg = self
        while reg is not None:
            ts.append(reg.t.tolist())
            reg = reg.super_region
        ts = list(reversed(ts))
        return str(ts)

    def num_predictions(self):
        tot_corr, tot_inc = 0, 0
        for _, T in self.sub_regions.iteritems():
            corr, inc = T.num_predictions()
            tot_corr += corr
            tot_inc += inc
        return tot_corr, tot_inc

    def get_predictions(self):
        preds = []
        for T in self.sub_regions:
            preds.extend(T.get_predictions)
        return preds

    def is_pure(self):
        first = True
        targ = None
        for _, T in self.sub_regions.iteritems():
            is_pure, target = T.is_pure()
            if not is_pure:
                return False, None
            if first:
                first = False
                targ = target
            elif targ != target:
                return False, None
        return True, target

    def concatenated_T(self):
        """Concatenate the set of binary tuples that describe this region
        and return the result:
        
        :return The binary tuples, concatenated to form a single numpy array"""
        ts = []
        current_reg = self
        while current_reg is not None:
            ts.append(current_reg.t)
            current_reg = current_reg.super_region
        ts = list(reversed(ts))
        as_array = np.concatenate(ts)
        return as_array


class FinalRegion(Region):

    def __init__(self, l, t, super_region):
        Region.__init__(self, l, t, super_region)
        self.predictions = []

    def num_predictions(self):
        correct = sum(int(p.target == p.predicted) for p in self.predictions)
        incorrect = len(self.predictions) - correct
        return correct, incorrect

    def get_predictions(self):
        return self.predictions

    def has_correct_pred(self):
        return any(map(lambda p: p.is_correct(), self.predictions))

    def has_inc_pred(self):
        return any(map(lambda p: not p.is_correct(), self.predictions))

    def is_pure(self):
        target = self.predictions[0].target
        for pred in self.predictions:
            if pred.target != target:
                return False, None
        return True, target

    def _true_for(self, predicate):
        for pred in self.predictions:
            if predicate(pred):
                return True
        return False

class Prediction:

    def __init__(self, target, predicted):
        self.target = target
        self.predicted = predicted

    def __str__(self):
        return "Target: "+str(self.target)+", Prediction: "+str(self.predicted)

    def is_correct(self):
        return self.target == self.predicted

class FreqTracker:

    def __init__(self):
        self.freq_perfect = 0
        self.freq_mixed = 0

