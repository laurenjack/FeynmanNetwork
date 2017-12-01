from train import *
from visualisation import *
from k_nearest import KNearest
import numpy as np


def _form_class_cluster(train_region_set):
    co_act_tups = []
    train_points = np.array(train_region_set.Ts)
    train_targets = train_region_set.all_targets
    for i in xrange(10):
        inds_of_class_i = np.argwhere(train_targets == i)[:, 0]
        co_act_tups.append(train_points[inds_of_class_i])
    return co_act_tups

def run(network, train_regions, conf, sess, test_samples=10):
    """This script is used to give a person and intuitive idea of what the KNearest neighbours
    to a given prediction looks like."""
    X = conf.X
    Y = conf.Y
    X_val = conf.X_val
    Y_val = conf.Y_val
    np.random.shuffle(X_val)
    np.random.shuffle(Y_val)

    # test the adverserial examples
    n = X.shape[0]
    val_n = X_val.shape[0]
    rand_inds = np.random.choice(np.arange(n), size=test_samples, replace=False)
    X_sub = X[rand_inds]
    adv_Y = Y[rand_inds]
    adv_X = gen_adverserial_examples(network, X_sub, adv_Y, conf, sess)
    # Create a region forest for the adverserial examples
    adv_region_forest = report(network, adv_X, adv_Y, conf, sess)
    adv_regions = adv_region_forest.all_final_regions()
    #show_original_vs_adv(X_sub, adv_X)


    # Find the regions for the validation set
    val_region_forest = report(network, X_val, Y_val, conf, sess)

    # Find the regions for random noise
    # X_noise = np.random.randn(test_samples, 784)
    # Y_noise = np.zeros((test_samples, 10))
    # garbage_forest = report(network, X_noise, Y_noise, conf, sess)
    # garbage_regions = garbage_forest.all_final_regions()


    # Find equal part correctly classified and incorrectly classified examples from the training set
    corr, incorr = val_region_forest.get_n_correct_and_incorrect(test_samples)
    k_nearest = KNearest(conf)

    class_orgd_act_tups = _form_class_cluster(train_regions)
    tc = k_nearest.train_counter
    tc_fd = {t: cc for t, cc in zip(tc.class_clusters, class_orgd_act_tups)}

    def centre_distance_for_all(final_reg_set):
        """Returns a map of targets to numpy arrays where each numpy array is the distance
        of the given instance from the centre of each class"""
        final_regions = final_reg_set.final_regions
        pred_to_centre_dist = {i: ([], []) for i in xrange(10)}
        Ts = final_reg_set.Ts
        for i in xrange(len(final_regions)):
            first_pred = final_regions[i].predictions[0]
            test_point = Ts[i]
            centre_dists = k_nearest.report_centre_distances(sess, tc_fd, test_point)
            dists, targs = pred_to_centre_dist[first_pred.predicted]
            dists.append(centre_dists)
            targs.append(first_pred.target)
        return pred_to_centre_dist

    def K_nearest_for_all(final_reg_set):
        final_regions = final_reg_set.final_regions
        Ts = final_reg_set.Ts
        for i in xrange(len(final_regions)):
            first_pred = final_regions[i].predictions[0]
            T = Ts[i]
            nearest, nearest_regions, distances, avg_dist_pc, count_dist = k_nearest.report_KNearest(sess, T, first_pred.predicted,
                                                                                train_regions)
            show_neighbouring_instances(nearest, first_pred.predicted, first_pred.target, nearest_regions,
                                        distances, avg_dist_pc ,count_dist)


    cd_corr = centre_distance_for_all(corr)
    cd_adv = centre_distance_for_all(adv_regions)

    def _print_results(dists_targs):
        dists, targs = dists_targs
        for dist, targ in zip(dists, targs):
            print "Target: "+str(targ)+"   Distances: "+str(dist)

    # Print the results
    for i in xrange(10):
        print "Predicted: "+str(i)
        print "Correct Validation:"
        _print_results(cd_corr[i])
        print "Adverserial:"
        _print_results(cd_adv[i])
        print ""



    # #K_nearest_for_all(corr)
    # # K_nearest_for_all(incorr)
    # print "ADVERSERIES"
    # K_nearest_for_all(adv_regions)
