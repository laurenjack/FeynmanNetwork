from tensorflow.examples.tutorials.mnist import input_data


class Config:

    def __init__(self, sizes, learning_rate, m, epochs, feyn_lr=0.01, feyn_epochs=20, k=10, epsilon=0.25, is_binary=True, is_w_pixels=False, is_rbf=False):
        self.sizes = sizes
        self.learning_rate = learning_rate
        self.m = m
        self.epochs = epochs
        self.feyn_lr = feyn_lr
        self.feyn_epochs = feyn_epochs
        self.k = k
        self.epsilon = epsilon
        self.is_binary = is_binary
        self.is_w_pixels = is_w_pixels
        self.is_rbf = is_rbf
        self.NETWORK_GLOBAL = 0

        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        train_set = mnist.train
        val_set = mnist.validation
        self.X_val = val_set.images#[:2000]
        self.Y_val = val_set.labels#[:2000]
        self.X = train_set.images#[:2000]
        self.Y = train_set.labels#[:2000]

class ConfBuilder:

    def __init__(self):
        self.is_rbf_soft = False

    def build(self):
        return ResConfig(self.num_classes, self.image_dims, self.train_dir, self.n,
            self.lr, self.lr_reduce_factor, self.lr_reduce_steps, self.momentum, self.m, self.reporting_m,
            self.max_steps, self.num_filter_list, self.num_block_list, self.pp_k_size, self.pp_stride, self.has_mp,
                         self.k_size, self.f_stride, self.stride, self.adv_epsilon, self.k, self.s, self.is_rbf_soft)


class ResConfig:

    def __init__(self, num_classes, image_dims, train_dir, n,
                 lr, lr_reduce_factor, lr_reduce_steps, momentum, m, reporting_m, max_steps, num_filter_list,
                 num_block_list, pp_k_size, pp_stride, has_mp, k_size,
                 f_stride, stride, adv_epsilon, k, s, is_rbf_soft):
        if len(num_filter_list) - 1 != len(num_block_list):
            raise ValueError("First num block is explicitly specified, so need one less element than num_filter_list")

        self.num_classes = num_classes
        self.image_dims = image_dims
        self.train_dir = train_dir
        self.n = n

        self.lr = lr
        self.lr_reduce_factor = lr_reduce_factor
        self.lr_reduce_steps = lr_reduce_steps
        self.momentum = momentum
        self.m = m
        self.reporting_m = reporting_m
        self.max_steps = max_steps

        self.pp_d = num_filter_list[0]
        self.pp_k_size = pp_k_size
        self.pp_stride = pp_stride
        self.has_mp = has_mp
        self.stacks = []
        for num_filter, num_block in zip(num_filter_list[1:], num_block_list):
            stack = Stack(num_block, num_filter, k_size, f_stride, stride)
            self.stacks.append(stack)

        self.adv_epsilon = adv_epsilon
        self.k = k
        self.s = s
        self.is_rbf_soft = is_rbf_soft



class Stack:
    """Configuration for an individual stack of a Resnet
    A stack is made up of one or more identical blocks"""

    def __init__(self, num_blocks, in_d, k_size, f_stride, stride):
        self.num_blocks = num_blocks
        self.in_d = in_d
        self.k_size = k_size
        self.f_stride = f_stride
        self.stride = stride

