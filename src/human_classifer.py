from train import *
from visualisation import *
from k_nearest import KNearest

def run(network, train_regions, conf, sess, test_samples=10):
    """This script is used to give a person and intuitive idea of what the KNearest neighbours
    to a given prediction looks like."""
    X = conf.X
    Y = conf.Y
    X_val = conf.X_val
    Y_val = conf.Y_val

    # test the adverserial examples
    n = X.shape[0]
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

    def K_nearest_for_all(final_reg_set):
        final_regions = final_reg_set.final_regions
        Ts = final_reg_set.Ts
        for i in xrange(len(final_regions)):
            first_pred = final_regions[i].predictions[0]
            T = Ts[i]
            nearest, nearest_regions, distances, avg_dist_pc = k_nearest.report_KNearest(sess, T, first_pred.predicted,
                                                                                train_regions)
            show_neighbouring_instances(nearest, first_pred.predicted, first_pred.target, nearest_regions,
                                        distances, avg_dist_pc)

    K_nearest_for_all(adv_regions)
    K_nearest_for_all(corr)
    K_nearest_for_all(incorr)
