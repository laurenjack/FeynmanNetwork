from k_nearest import *
from linear_regions import FinalRegion
from linear_regions import FinalRegionSet
from linear_regions import Prediction
from unittest import TestCase
import numpy as np

class KNearestSpec(TestCase):

    def test_k_nearest(self):
        pred_reg, pred_class, final_reg_set, conf = self._setup()

        #Specify expected output
        exp_nearest = {4: InstanceTracker(), 5: InstanceTracker(), 6: InstanceTracker()}
        exp_nearest[4].correct = 2
        exp_nearest[4].incorrect = 1
        exp_nearest[5].correct = 1
        exp_nearest[6].incorrect = 1

        #Build KNearest object
        kNearest = KNearest(conf)

        #Compute K-Nearest
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        act_nearest, _, _, _ = kNearest.report_KNearest(sess, pred_reg, pred_class, final_reg_set)

        #Compare
        self.assertEqual(exp_nearest, act_nearest)

    def test_simple_stat(self):
        pred_reg, pred_class, final_reg_set, conf = self._setup()

        #Build KNearest object
        kNearest = KNearest(conf)

        #Compute simple statstics
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        simple_stats = kNearest.report_simple_stats(sess, pred_reg, pred_class, final_reg_set)

        #Compare
        self.assertAlmostEqual(2.5, simple_stats.avg_dist)
        self.assertAlmostEqual(0.4, simple_stats.pred_and_correct)
        self.assertAlmostEqual(0.44, simple_stats.inv_pred_entropy)
        self.assertAlmostEqual(0.4, simple_stats.incorrect_neighbours)

    def _setup(self):
        pred_reg = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        pred_class = 4
        # Create the final regions, only including the relevant properties
        final_regs = [FinalRegion(None, None, None), FinalRegion(None, None, None),
                      FinalRegion(None, None, None), FinalRegion(None, None, None),
                      FinalRegion(None, None, None)]
        final_regs[0].predictions = [Prediction(4, 4), Prediction(4, 3), Prediction(4, 4)]  # Nearby 1
        final_regs[1].predictions = [Prediction(4, 4)]
        final_regs[2].predictions = [Prediction(4, 4), Prediction(4, 4)]
        final_regs[3].predictions = [Prediction(5, 5), Prediction(6, 5)]  # Nearby 2
        final_regs[4].predictions = [Prediction(4, 4), Prediction(4, 3), Prediction(4, 4)]
        # Create the corresponding full tuple set
        Ts = np.zeros((5, 8))
        Ts[0] = np.array([1, 1, 1, 0, 1, 0, 1, 1])  # dist = 2
        Ts[1] = np.array([0, 1, 0, 1, 0, 1, 0, 1])  # dist = 8
        Ts[2] = np.array([1, 1, 0, 0, 0, 1, 0, 0])  # dist = 5
        Ts[3] = np.array([0, 1, 0, 0, 1, 0, 1, 0])  # dist = 3
        Ts[4] = np.array([1, 0, 0, 1, 1, 1, 0, 0])  # dist = 4
        final_reg_set = FinalRegionSet(final_regs, Ts)

        #Create corresponding configuration
        class Conf:
            pass
        conf = Conf()
        conf.sizes = [None, 2, 2, 2, 2, None]
        conf.k = 2

        return pred_reg, pred_class, final_reg_set, conf

