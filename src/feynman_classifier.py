import tensorflow as tf
import numpy as np
from k_nearest import KNearest
from train import *
from feynmanmodules.simple_logistic import SimpleLogistic

def run(training_region_forest, val_region_forest, conf, feyn_n=1000):
    """Trains a Feynman module to predict the probability its corresponding
    network is correct, and reports the results."""
    # X = conf.X
    # Y = conf.Y
    m = conf.m
    epochs = conf.feyn_epochs

    # Find equal part correctly classified and incorrectly classified examples from the training set
    kNearest = KNearest(conf)
    simpleLogistic = SimpleLogistic(conf)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    corr, incorr = val_region_forest.get_n_correct_and_incorrect(feyn_n)
    train_region_set = training_region_forest.all_final_regions()
    X_feyn, Y_feyn = _create_merged_ds(corr, incorr, train_region_set, kNearest, sess)
    #Standardize the distances so the network can train TODO consider log distance
    X_feyn[:,0] = _normalize(X_feyn[:,0])

    #Create a simple Feyman Module and train it
    train(simpleLogistic, X_feyn, Y_feyn, m, epochs, sess, sub_set_report=40, is_feyn=True)


def _create_merged_ds(corr, incorr, all_regions_forest, kNearest, sess):
    X_corr, Y_corr = _create_data_feynman(corr, all_regions_forest, kNearest, sess)
    X_inc, Y_inc = _create_data_feynman(incorr, all_regions_forest, kNearest, sess)
    return np.concatenate((X_corr, X_inc)), np.concatenate((Y_corr, Y_inc))

def _create_data_feynman(prediction_set, all_regions_set, kNearest, sess):
    """Given a particular set of predictions, produce the simple statistics required for training/prediction
    in a Feynman module"""
    predicted_regions = prediction_set.final_regions
    Ts = prediction_set.Ts
    xs = []
    ys = []
    for i in xrange(len(predicted_regions)):
        print "Prediction: " + str(i) + " complete"
        first_pred = predicted_regions[i].predictions[0]
        T = Ts[i]
        simpleStats = kNearest.report_simple_stats(sess, T, first_pred.predicted, all_regions_set)
        x = simpleStats.as_vector()
        #Create the target
        y = np.zeros((1, 2))
        if first_pred.is_correct():
            y[0,0] = 1
        else:
            y[0,1] = 1
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)

def _normalize(arr):
    n = arr.shape[0]
    mean = np.mean(arr)
    sd = (np.sum((arr - mean) ** 2.0)/ float(n)) ** 0.5
    return (arr - mean)/sd

