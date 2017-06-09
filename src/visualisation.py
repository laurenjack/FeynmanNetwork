import numpy as np
import matplotlib.pyplot as plt

def frequency_of_instance_per_region(num_pred_map):
    num_preds, perfect, mixed, max_pred = _as_arrays(num_pred_map)
    ind = np.arange(max_pred)

    plt.bar(ind, perfect, width=0.2, color='g')
    plt.bar(ind, mixed, width=0.2, color='r')
    plt.xticks(ind, num_preds)
    plt.show()

def show_neighbouring_instances(nearest_instances, predicted, target, avg_sim):
    #Convert to numpy data structures
    num_bars = len(nearest_instances)
    ind = np.arange(num_bars)
    targets, num_correct, num_incorrect = np.zeros(num_bars), np.zeros(num_bars), np.zeros(num_bars)
    i = 0
    for key, it in nearest_instances.iteritems():
        targets[i] = key
        num_correct[i] = it.correct
        num_incorrect[i] = it.incorrect
        i += 1

    #Plot as bar charts
    plt.title('Predicted: '+str(predicted), loc='left')
    plt.title('Sim: '+str(avg_sim))
    plt.title('Target: '+str(target), loc='right')
    plt.bar(ind, num_correct, width=0.2, color='g')
    plt.bar(ind, num_incorrect, width=0.2, color='r')
    plt.xticks(ind, targets)
    plt.show()

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




