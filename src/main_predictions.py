from train import *
from network import FeedForward
from conf import Config
import feynman_classifier
import human_classifer

np.set_printoptions(precision=2, suppress=True)

sizes = [784, 784, 784, 10]
learning_rate = 0.01
m = 20
epochs = 25
conf = Config(sizes, learning_rate, m, epochs, feyn_lr=0.05, feyn_epochs=100, k=15, epsilon=0.05, is_binary=True, is_w_pixels=False)#/255.0)
network = FeedForward(conf)
X = conf.X
Y = conf.Y

#Train the network
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
train(network, X, Y, m, epochs, sess)

#Load validation set
X_val = conf.X_val
Y_val = conf.Y_val
batch_inds = np.arange(X_val.shape[0])
np.random.shuffle(batch_inds)
X_val = X_val[batch_inds]
Y_val = Y_val[batch_inds]

#Select a sub samples for validation and adverserial prob testing
X_adv = X_val[0:10]
Y_adv = Y_val[0:10]
X_adv = gen_adverserial_examples(network, X, Y, conf, sess)
X_val_sub = X_val[10:20]
Y_val_sub = Y_val[10:20]

#Report the validation predictions, then adverserial
report_predictions(network, X_val_sub, Y_val_sub, conf, sess)
print ""
report_predictions(network, X_adv, Y_adv, conf, sess)



