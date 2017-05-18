import numpy as np

class RegionSetBuilder:

    def __init__(self, conf):
        #Create map to store regions by layer
        self.sizes = conf.sizes
        num_layers = len(self.sizes) - 2
        self.layer_map = {}
        for l in xrange(1, num_layers+1):
            #Create set to hold all t's (the tuple id) that occur at that layer
            self.layer_map[l] = {}


    def putRegion(self, T_by_layer):
        #Re-organise ts according to instances rather than layers
        all_T = self._transpose(T_by_layer)

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
                t = T_by_layer[b][l]
                T.append(t)
            all_T.append(T)
        return all_T

    # def _create_hash_powers(self, sizes):
    #     """Create the powers that each index i of t is raised to, so that each t may be represented
    #     by a unique integer"""
    #     hash_powers = []
    #     for l in sizes[1:-1]:
    #         hp = np.zeros(shape=l, dtype=np.long)
    #         for i in xrange(l):
    #             np.zeros[i] = 2 << i
    #         hash_powers.append(hp)
    #     return hash_powers






class Region:
    """Data structure representing a linear region at a given layer"""
    def __init__(self):
        self.instances = {}
        self.layer = None
        self.t = None
        self.sub_regions = []
        self.super_region = None

    def __eq__(self, other):
        if not isinstance(other, Region):
            return False
        # if self.layer is None or self.id is None:
        #     raise TypeError("")
        if self.layer != other.layer:
            return False
        if self. t!= other.t:
            return False
        if self.super_region != other.super_region:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(self, other)







class Instance:

    def __init__(self):
        self.correct = 0
        self.incorrect = 0
