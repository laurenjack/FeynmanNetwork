from train import *
from network import FeedForward
from conf import Config
import feynman_classifier
import human_classifer
import visualisation as vis
import intersection_diagnostics as ri

np.set_printoptions(precision=2, suppress=True)

def f_layer_params(net, session):
    init_W, init_x_bar = net.get_final_layer_params()
    init_W = session.run(init_W)
    init_x_bar = session.run(init_x_bar)
    return init_W, init_x_bar

class Counter:

    def __init__(self):
        self.count = 0

    def inc(self):
        self.count += 1

NET_GLOBAL = Counter()
sizes = [784, 500, 10]
learning_rate = 0.003
m = 20
epochs = 20
num_adv = 20
conf = Config(sizes, learning_rate, m, epochs, feyn_lr=0.05, feyn_epochs=100, k=15, epsilon=0.25, is_binary=True, is_w_pixels=False, is_rbf=True)#/255.0)
network = FeedForward(conf, NET_GLOBAL)
X = conf.X
Y = conf.Y

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# See the initital rbf wieghts and x_bars
init_W, init_x_bar = f_layer_params(network, sess)

#Train the network
train(network, X, Y, m, epochs, sess)

# See the final rbf wieghts and x_bars
final_W, final_x_bar = f_layer_params(network, sess)

#Report the highest intersection of rbfs
ri.x_bar_vs_centre(X, Y, final_x_bar, final_W, sess, network)
rbf1, rbf2 = ri.find_pairwise_intersections(network, 200, 0.1, 0.1, sess)
print "RBF intersection max: "+str(rbf1)+"   "+str(rbf2)



#Load validation set
X_val = conf.X_val
Y_val = conf.Y_val
batch_inds = np.arange(X_val.shape[0])
np.random.shuffle(batch_inds)
X_val = X_val[batch_inds]
Y_val = Y_val[batch_inds]

#Select a sub samples for validation and adverserial prob testing
X_orig = X_val[0:num_adv]
Y_orig = Y_val[0:num_adv]
# X_orig = np.concatenate(np.array([X_orig] * 10, copy=True).transpose(1, 0, 2), axis=0)
# #X_adv = gen_adverserial_examples(network, X_orig, Y_orig, conf, sess)
# y_other_inds = np.concatenate(np.array([np.arange(10)] * num_adv, copy=True), axis=0)
# #y_other_inds = (np.argmax(Y_orig, axis=1) + 2) % 10
# y_other = np.zeros((Y_orig.shape[0] * 10, Y_orig.shape[1]))
# y_other[np.arange(y_other_inds.shape[0]), y_other_inds] = 1.0
#
# X_adv = to_other_class_adverseries(network, X_orig, y_other, sess)
X_val_sub = X_val[num_adv:num_adv*2]
Y_val_sub = Y_val[num_adv:num_adv*2]

#Report the validation predictions, then adverserial
report_predictions(network, X_val_sub, Y_val_sub, conf, sess)
print ""
#repeated_targets = np.concatenate(np.array([Y_orig] * 10, copy=True).transpose(1, 0, 2), axis=0)
X_adv, X_sub, y_adv = gen_black_box_adverseries(num_adv, NET_GLOBAL)
report_predictions(network, X_adv, y_adv, conf, sess)
vis.show_original_vs_adv(X_sub, X_adv)






