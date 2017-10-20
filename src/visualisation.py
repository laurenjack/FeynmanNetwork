import numpy as np
import matplotlib.pyplot as plt
from k_nearest import InstanceTracker

def frequency_of_instance_per_region(num_pred_map):
    num_preds, perfect, mixed, max_pred = _as_arrays(num_pred_map)
    ind = np.arange(max_pred)

    plt.bar(ind, perfect, width=0.2, color='g')
    plt.bar(ind, mixed, width=0.2, color='r')
    plt.xticks(ind, num_preds)
    plt.show()

def show_original_vs_adv(originals, adverseries):
    """Take two numpy arrays where the first an array of original images from
    a data set and the second is a corresponding array of adverserial examples
    generated from the correspondent original."""
    for i in xrange(originals.shape[0]):
        org = originals[i]
        adv = adverseries[i]
        _plot_image(org, 1)
        _plot_image(adv, 2)
        plt.show()


def _plot_image(im_vector, sub_num):
    sub_plot = plt.subplot(1, 2, sub_num)
    sub_plot.yaxis.set_visible(False)
    sub_plot.xaxis.set_visible(False)
    sub_plot.imshow(im_vector.reshape(28, 28), interpolation='nearest')


def show_neighbouring_instances_conv(kn_result):
    #Creat dictionary of targets to count of correct/incorrect
    nearest_targets = kn_result.nearest_targets
    k = nearest_targets.shape[0]
    nearest_map = {}
    for i in xrange(k):
        target = nearest_targets[i]
        if target not in nearest_map.keys():
            it = InstanceTracker()
            nearest_map[target] = it
        else:
            it = nearest_map[target]
        was_correct = kn_result.were_they_correct[i]
        it.increment(was_correct)

    target = kn_result.target
    prediction = kn_result.prediction
    avg_distance = kn_result.avg_distance
    _print_neighbours(k, kn_result)
    _display_kn_bars(nearest_map, target, prediction, avg_distance)



def show_neighbouring_instances(nearest_instances, predicted, target, nearest_regions, distances, avg_dist_pc):
    # Print the details fo each nearby region
    _print_regions(nearest_regions, distances)
    _display_kn_bars(nearest_instances, target, predicted, avg_dist_pc)

def _display_kn_bars(nearest_map, target, predicted, avg_dist):
    # Convert to numpy data structures
    num_bars = len(nearest_map)
    ind = np.arange(num_bars)
    targets, num_correct, num_incorrect = np.zeros(num_bars), np.zeros(num_bars), np.zeros(num_bars)
    i = 0
    for key, it in nearest_map.iteritems():
        targets[i] = key
        num_correct[i] = it.correct
        num_incorrect[i] = it.incorrect
        i += 1

        # Plot as bar charts
    plt.title('Predicted: ' + str(predicted), loc='left')
    plt.title('Dist: ' + str(avg_dist))
    plt.title('Target: ' + str(target), loc='right')
    plt.bar(ind, num_correct, width=0.2, color='g')
    plt.bar(ind, num_incorrect, width=0.2, color='r')
    plt.xticks(ind, targets)
    plt.show()


def _print_regions(nearest_regions, similarilties):
    """Print the nearest neighbours in order of how far they are from the current prediction"""
    print ""
    for i in xrange(len(nearest_regions)):
        reg = nearest_regions[i]
        sim = similarilties[i]
        print "Region: "+str(i+1)+"   ",
        for pred in reg.predictions:
            print str(pred)+", Sim: "+str(sim)+"   ",
        print "\n"


def _print_neighbours(k, kn_result):
    targets = kn_result.nearest_targets
    predictions = kn_result.nearest_predictions
    distances = kn_result.their_distances
    for i in xrange(k):
        t = targets[i]
        p = predictions[i]
        d = distances[i]
        print "Target: "+str(t)+", Prediction: "+str(p)+", Distance: "+str(d)
        print "\n"




def _as_arrays(num_pred_map):
    pred_keys = num_pred_map.keys()
    max_pred = max(pred_keys)
    num_preds = np.arange(1, max_pred+1)
    perfect = np.zeros(max_pred)
    mixed = np.zeros(max_pred)
    for i in xrange(max_pred):
        num_pred = num_preds[i]
        if num_pred in pred_keys:
            freq_tracker = num_pred_map[num_pred]
            perfect[i] = freq_tracker.freq_perfect
            mixed[i] = freq_tracker.freq_mixed

    return num_preds, perfect, mixed, max_pred




