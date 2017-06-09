
from linear_regions import *
from unittest import TestCase
import numpy as np

class RegionSetBuilderSpec(TestCase):

    def test_put_regions(self):
        _, exp_roots, act_forest = self._create_test_forest()
        act_roots = act_forest.roots
        self.assertEqual(len(exp_roots), len(act_roots))
        for key in exp_roots.keys():
            self.assertEqual(exp_roots[key], act_roots[key])
        self._assert_regions_equal(exp_roots, act_roots)


    def test_all_final(self):
        # Make the region forest, and the expected final regions
        exp_final_regions, _, region_forest = self._create_test_forest()
        #Make the expected T's
        exp_Ts = [np.array([1, 1, 1, 1, 0, 1, 0]),
                  np.array([1, 1, 1, 1, 0, 1, 1]),
                  np.array([1, 1, 1, 1, 0, 0, 1]),
                  np.array([1, 1, 0, 1, 1, 1, 1]),
                  np.array([1, 1, 0, 1, 0, 1, 1]),
                  np.array([1, 0, 1, 0, 1, 1, 0])]

        # Test if all_final_regions works, by checking:
        # 1) The final regions are as expected
        final_region_set = region_forest.all_final_regions()
        act_final_regions = final_region_set.final_regions
        self._is_pertubation_of(exp_final_regions, act_final_regions)
        # 2) The concatenated T sets are as expected
        act_Ts = final_region_set.Ts.tolist()
        self._is_pertubation_of_nd(exp_Ts, act_Ts)

    def test_get_n_correct_and_incorrect(self):
        exp_corr = [np.array([1, 0, 1, 0, 1, 1, 0]),
                     np.array([1, 1, 0, 1, 0, 1, 1])]
        exp_inc = [np.array([1, 1, 0, 1, 1, 1, 1]),
                     np.array([1, 1, 1, 1, 0, 0, 1])]

        _, _, act_forest = self._create_test_forest()
        act_corr, act_inc = act_forest.get_n_correct_and_incorrect(2)
        act_corr, act_inc = act_corr.Ts, act_inc.Ts

        self.assertTrue(np.array_equal(exp_corr, act_corr))
        self.assertTrue(np.array_equal(exp_inc, act_inc))

    def _assert_regions_equal(self, exp, act):
        self._is_pertubation_of(exp.keys(), act.keys())
        if self._is_all_final(exp):
            return
        else:
            for key in exp.keys():
                exp_subs = exp[key].sub_regions
                act_subs = exp[key].sub_regions
                self._assert_regions_equal(exp_subs, act_subs)

    def _is_all_final(self, regions):
        final = False
        not_final = False
        for reg in regions.values():
            if isinstance(reg, FinalRegion):
                final = True
            else:
                not_final = True
        if final and not not_final:
            return True
        if not_final and not final:
            return False
        raise Exception("Should be all final or not all final nothing in between")

    def _is_pertubation_of(self, l1, l2):
        """Assert two list are permutations of each other"""
        self.assertEquals(len(l1), len(l2))
        for e in l1:
            self.assertTrue(e in l2)
        for e in l2:
            self.assertTrue(e in l1)

    def _is_pertubation_of_nd(self, l1, l2):
        """Assert two list of ndarrays are permutations of each other"""
        self.assertEquals(len(l1), len(l2))
        for e in l1:
            self.assertTrue(self._is_in(e, l2))
        for e in l2:
            self.assertTrue(self._is_in(e, l1))

    def _is_in(self, element, ndarrays):
        for arr in ndarrays:
            if np.array_equal(element, arr):
                return True
        return False

    def _not_in_message(self, e, list):
        print str(e) + " is not in: ".join(map(str, list))

    def _create_test_forest(self):
        """Create a fresh region forest for testing"""
        # Create input data
        sizes = [None, 3, 2, 2, None]
        all_T = [None] * 4
        all_targ = [None] * 4
        all_pred = [None] * 4

        all_T[0] = [np.array([[1, 1, 1], [1, 1, 1]]),
                    np.array([[1, 0], [1, 0]]),
                    np.array([[1, 0], [1, 1]])]
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

        # Create roots
        exp_roots = {7: Region(0, np.array([1, 1, 1]), None),
                     6: Region(0, np.array([1, 1, 0]), None),
                     5: Region(0, np.array([1, 0, 1]), None)}

        # Layer 1
        first = exp_roots[7]
        first.sub_regions = {2: Region(1, np.array([1, 0]), first)}

        second = exp_roots[6]
        second.sub_regions = {2: Region(1, np.array([1, 0]), second),
                              3: Region(1, np.array([1, 1]), second)}

        third = exp_roots[5]
        third.sub_regions = {1: Region(1, np.array([0, 1]), third)}

        exp_final_regions = []
        # Layer 2
        first_1 = first.sub_regions[2]
        final1 = FinalRegion(2, np.array([0, 1]), first_1)
        final2 = FinalRegion(2, np.array([1, 0]), first_1)
        final3 = FinalRegion(2, np.array([1, 1]), first_1)
        first_1.sub_regions = {1: final1, 2: final2, 3: final3}
        final1.predictions = [Prediction(7, 8), Prediction(7, 9), Prediction(7, 8)]
        final2.predictions = [Prediction(4, 4), Prediction(4, 4)]
        final3.predictions = [Prediction(4, 5)]
        exp_final_regions.extend([final1, final2, final3])

        second_1 = second.sub_regions[2]
        final1 = FinalRegion(2, np.array([1, 1]), second_1)
        second_1.sub_regions = {3: final1}
        final1.predictions = [Prediction(6, 6)]
        exp_final_regions.append(final1)

        second_2 = second.sub_regions[3]
        final1 = FinalRegion(2, np.array([1, 1]), second_2)
        second_2.sub_regions = {3: final1}
        final1.predictions = [Prediction(2, 3)]
        exp_final_regions.append(final1)

        third_1 = third.sub_regions[1]
        final1 = FinalRegion(2, np.array([1, 0]), third_1)
        third_1.sub_regions = {2: final1}
        final1.predictions = [Prediction(1, 1)]
        exp_final_regions.append(final1)

        # Create the region set builder
        class Conf:
            pass

        conf = Conf()
        conf.sizes = sizes
        rsb = RegionSetBuilder(conf)

        # Put each batch through the region builder
        for T, targets, predictions in zip(all_T, all_targ, all_pred):
            rsb.putRegion(T, targets, predictions)

        # Verify each root produced is equal to the expected roots
        act_forest = rsb.get_forest()

        return exp_final_regions, exp_roots, act_forest



