import numpy as np
import tensorflow as tf

class Reporter:

    def __init__(self, conf, resnet):
        self.m = conf.m
        self.resnet = resnet

    def report_full_ds(self, all_images, all_targets, sess, sample_size):
        """Given a data set, feed"""
        #Iterate over batches and produce Prediction results
        batch_indicies = np.arange(sample_size)
        all_pre_final = np.zeros((sample_size, 1136), dtype=np.float32)
        all_predictions = np.zeros(sample_size)
        all_was_correct = np.zeros(sample_size, dtype=np.bool)
        all_ftws = np.zeros((sample_size, 1136), dtype=np.float32)

        pred = self.resnet.prediction_result()
        for k in xrange(0, sample_size, self.m):
            batch = batch_indicies[k:k + self.m]
            x = all_images[batch]
            y = all_targets[batch]
            pre_final, predictions, targets, was_correct =\
                sess.run(pred, {self.resnet.is_training: False, self.resnet.images: x, self.resnet.labels: y})
            all_pre_final[batch] = pre_final
            all_predictions[batch] = predictions
            all_was_correct[batch] = was_correct
            #all_ftws[batch] = ftws
        prediction_result = PredictionResult(all_pre_final, all_predictions, all_targets, all_was_correct)
        return prediction_result, all_ftws


    def _concat(self, prediction_results):
        all_pre_final = []
        all_predictions = []
        all_targets = []
        all_was_correct = []
        for pr in prediction_results:
            all_pre_final.append(pr.pre_final)
            all_predictions.append(pr.predictions)
            all_targets.append(pr.targets)
            all_was_correct.append(pr.was_correct)
        all_pre_final = np.concatenate(all_pre_final)
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        all_was_correct = np.concatenate(all_was_correct)
        return PredictionResult(all_pre_final, all_predictions, all_targets, all_was_correct)

class PredictionResult:
    """Represents a batch of predictions made by a neural network"""

    def __init__(self, pre_final, predictions, targets, was_correct):
        self.pre_final = pre_final
        self.predictions = predictions
        self.targets = targets
        self.was_correct = was_correct





