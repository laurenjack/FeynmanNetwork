import numpy as np
from linear_regions import *
from unittest import TestCase

class RegionSetBuilderSpec(TestCase):

    def test_put_regions(self):
        #Create input data
        sizes = [None, 3, 2, 2, None]
        all_T = [None] * 4
        all_targ = [None] * 4
        all_pred = [None] * 4

        all_T[0] = [np.array([[1, 1, 1], [1, 1, 1]]),
             np.array([[1, 0], [1,0]]),
                    np.array([[1, 0], [1,1]])]
        all_targ[0] = np.array([4, 4])
        all_pred[0] = np.array([4, 5])

        all_T[1] = [np.array([[1, 1, 0], [1, 1, 1]]),
                    np.array([[1, 0], [1, 0]]),
                    np.array([[1, 1], [1, 0]])]
        all_targ[1] = np.array([6, 4])
        all_pred[1] = np.array([6, 4])

        all_T[2] = [np.array([[1, 1, 1], [1, 1, 0]]),
                    np.array([[1, 0], [1, 1]]),
                    np.array([[0, 1], [1, 1]])]
        all_targ[2] = np.array([7, 2])
        all_pred[2] = np.array([8, 3])

        all_T[3] = [np.array([[1, 1, 1], [1, 1, 1], [1, 0, 1]]),
                    np.array([[1, 0], [1, 0], [0, 1]]),
                    np.array([[0, 1], [0, 1], [1, 0]])]
        all_targ[3] = np.array([7, 7, 1])
        all_pred[3] = np.array([9, 8, 1])

        #Create roots
        exp_roots = {7: Region(0, np.array([1, 1, 1]), None),
                     6: Region(0, np.array([1, 1, 0]), None),
                     5: Region(0, np.array([1, 0, 1]), None)}

        # Layer 1
        first = exp_roots[7]
        first.sub_regions = {2: Region(1, np.array([1, 0]), first)}

        second = exp_roots[6]
        second.sub_regions = {2: Region(1, np.array([1, 0]), second),
                              3: Region(1, np.array([1,1]), second)}

        third = exp_roots[5]
        third.sub_regions = {1: Region(1, np.array([0,1]), third)}

        #Layer 2
        first_1 = first.sub_regions[2]
        final1 = FinalRegion(2, np.array([0,1]), first_1)
        final2 = FinalRegion(2, np.array([1,0]), first_1)
        final3 = FinalRegion(2, np.array([1,1]), first_1)
        first_1.sub_regions = {1: final1, 2: final2, 3: final3}
        final1.predictions = [Prediction(7, 8), Prediction(7, 9), Prediction(7, 8)]
        final2.predictions = [Prediction(4, 4), Prediction(4, 4)]
        final3.predictions = [Prediction(4, 5)]

        second_1 = second.sub_regions[2]
        final1 = FinalRegion(2, np.array([1,0]), second_1)
        second_1.sub_regions = {3: final1}
        final1.predictions = [Prediction(6, 6)]

        second_2 = second.sub_regions[3]
        final1 = FinalRegion(2, np.array([1, 1]), second_2)
        second_2.sub_regions = {3: final1}
        final1.predictions = [Prediction(2, 3)]

        third_1 = third.sub_regions[1]
        final1 = FinalRegion(2, np.array([1, 0]), third_1)
        third_1.sub_regions = {2: final1}
        final1.predictions = [Prediction(1, 1)]

        #Create the region set builder
        class Conf:
            pass
        conf = Conf()
        conf.sizes = sizes
        rsb = RegionSetBuilder(conf)


        #Put each batch through the region builder
        for T, targets, predictions in zip(all_T, all_targ, all_pred):
            rsb.putRegion(T, targets, predictions)


        #Verify each root produced is equal to the expected roots
        act_roots = rsb.get_forest().roots
        self.assertEqual(len(exp_roots), len(act_roots))
        for key in exp_roots.keys():
            self.assertEqual(exp_roots[key], act_roots[key])
