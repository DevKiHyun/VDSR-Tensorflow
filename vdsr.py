import tensorflow as tf
import numpy as np

class VDSR:
    def __init__(self, config):
        self.n_channel = config.n_channel
        self.weights = {}
        self.biases = {}
        self.global_step = tf.Variable(0, trainable=False)
        self.psnr = tf.Variable(0, dtype=tf.float32, trainable=False)
        self.X = tf.placeholder(tf.float32, shape=[None, None, None, self.n_channel])
        self.Y = tf.placeholder(tf.float32, shape=[None, None, None, self.n_channel])

    def _conv2d_layer(self, inputs, filters_size, stddev, strides=[1,1], add_bias=False, name=None,
                      padding="SAME", activation=None):
        filters = self._get_conv_filters(filters_size, name, stddev)
        strides = [1, *strides, 1]

        conv_layer = tf.nn.conv2d(inputs, filters, strides=strides, padding=padding, name=name + "_layer")

        if add_bias != False:
            conv_layer = tf.add(conv_layer, self._get_bias(filters_size[-1], name))
        if activation != None:
            conv_layer = activation(conv_layer)

        return conv_layer

    def _get_conv_filters(self, filters_size, name, stddev):
        name = name+"_weights"
        #initializer = tf.contrib.layers.xavier_initializer()
        initializer = tf.random_normal
        conv_weights = tf.Variable(initializer(shape=filters_size, stddev=stddev), name=name)
        self.weights[name] = conv_weights

        return conv_weights

    def _get_bias(self, bias_size, name):
        name = name+"_bias"
        bias = tf.Variable(tf.zeros([bias_size]), name=name)
        self.biases[name] = bias

        return bias

    def _residual_block(self, inputs, name=None, activation=None):
        skip_connection = tf.identity(inputs, name="skip_connection_{}".format(name))

        residual_net = self._conv2d_layer(inputs=inputs, filters_size=[3, 3, 64, 64], stddev=np.sqrt(2.0 / 9 / 64),
                                    add_bias=True, name="residual_block_{}_1".format(name), activation=activation)
        residual_net = self._conv2d_layer(inputs=residual_net, filters_size=[3, 3, 64, 64], stddev=np.sqrt(2.0 / 9 / 64),
                                    add_bias=True, name="residual_block_{}_2".format(name), activation=activation)

        output = tf.add(residual_net, skip_connection)

        if activation != None:
            output = activation(output)

        return output

    def neuralnet(self):
        self.conv_net = self._conv2d_layer(self.X, filters_size=[3, 3, self.n_channel, 64], stddev=np.sqrt(2.0/9), add_bias=True, name="conv_0",
                                          activation=tf.nn.relu)
        for i in range(1,19):
            self.conv_net = self._conv2d_layer(self.conv_net, filters_size=[3, 3, 64, 64], stddev=np.sqrt(2.0/9/64), add_bias=True,
                                         name="conv_{}".format(i), activation=tf.nn.relu)

        self.conv_net = self._conv2d_layer(self.conv_net, filters_size=[3, 3, 64, self.n_channel],
                                           stddev=np.sqrt(2.0 / 9 / 64), add_bias=True, name="conv_19")
        self.conv_net = tf.add(self.conv_net, self.X)


    def optimize(self, learning_rate, grad_clip, on_grad_clipping):
        self.cost = tf.reduce_sum(tf.nn.l2_loss(tf.subtract(self.Y, self.conv_net)))

        # weight decay
        conv_weights = {}
        conv_weights.update(self.weights)
        conv_weights.update(self.biases)
        for conv_weight in list(conv_weights.keys()):
            self.cost += tf.nn.l2_loss(conv_weights[conv_weight] * (1e-4))

        USE_GRADIENT_CLIPPING = on_grad_clipping
        if USE_GRADIENT_CLIPPING == False:
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost, global_step=self.global_step)
        else:
            opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
            grads = opt.compute_gradients(self.cost)
            clip_value = tf.Variable(grad_clip / learning_rate)
            # opt_grads = [(tf.clip_by_value(grad, -clip_value, clip_value), var) for grad, var in grads]
            opt_grads = [(tf.clip_by_norm(grad, clip_value), var) for grad, var in grads]
            self.optimizer = opt.apply_gradients(opt_grads, global_step=self.global_step)

    def summary(self, learning_rate):
        '''
        for weight in list(self.weights.keys()):
            tf.summary.histogram(weight, self.weights[weight])
        for bias in list(self.biases.keys()):
            tf.summary.histogram(bias, self.biases[bias])
        '''

        #tf.summary.scalar('Loss', self.cost)
        tf.summary.scalar('Average test psnr', self.psnr)
        tf.summary.scalar('Learning rate', learning_rate)

        self.summaries = tf.summary.merge_all()
