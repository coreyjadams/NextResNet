import sys

import tensorflow as tf

from utils import residual_block, downsample_block

# Declaring exception names:
class ConfigurationException(Exception): pass
class IncompleteFeedDict(Exception): pass



# Main class
class resnet(object):
    '''Define a network model and run training

    U resnet implementation
    '''
    def __init__(self, params):
        '''initialization

        Requires a list of parameters as python dictionary

        Arguments:
            params {dict} -- Network parameters

        Raises:
            ConfigurationException -- Missing a required parameter
        '''
        required_params =[
            'MINIBATCH_SIZE',
            'SAVE_ITERATION',
            'NUM_LABELS',
            'N_INITIAL_FILTERS',
            'BATCH_NORMALIZATION',
            'NETWORK_DEPTH',
            'RESIDUAL_BLOCKS_PER_LAYER',
            'LOGDIR',
            'BASE_LEARNING_RATE',
            'TRAINING',
            'RESTORE',
            'TRAINING_ITERATIONS',
        ]

        for param in required_params:
            if param not in params:
                raise ConfigurationException("Missing parameter "+ str(param))

        self._params = params

    def construct_network(self, dims):
        '''Build the network model

        Initializes the tensorflow model according to the parameters
        '''

        tf.reset_default_graph()



        # Initialize the input layers:
        self._input_image  = tf.placeholder(tf.float32, dims, name="input_image")
        self._input_labels = tf.placeholder(tf.int64,
            [dims[0], self._params['NUM_LABELS']], name="input_image")


        logits = self._build_network(self._input_image)


        self._softmax = tf.nn.softmax(logits)
        self._predicted_labels = tf.argmax(logits, axis=-1)

        # Keep a list of trainable variables for minibatching:
        with tf.variable_scope('gradient_accumulation'):
            self._accum_vars = [tf.Variable(tv.initialized_value(),
                                trainable=False) for tv in tf.trainable_variables()]


        # Accuracy calculations:
        with tf.name_scope('accuracy'):
            self._total_accuracy   = tf.reduce_mean(tf.cast(
                tf.equal(self._predicted_labels,
                         tf.argmax(self._input_labels, axis=-1)), tf.float32))

            # Add the accuracies to the summary:
            tf.summary.scalar("Total_Accuracy", self._total_accuracy)

        # Loss calculations:
        with tf.name_scope('cross_entropy'):

            # Unreduced loss, shape [BATCH, L, W]
            losses = tf.nn.softmax_cross_entropy_with_logits(
                labels=self._input_labels, logits=logits)

            self._loss = tf.reduce_mean(losses)

            # Add the loss to the summary:
            tf.summary.scalar("Total_Loss", self._loss)


        # Optimizer:
        with tf.name_scope("training"):
            self._global_step = tf.Variable(0, dtype=tf.int32,
                trainable=False, name='global_step')
            if self._params['BASE_LEARNING_RATE'] <= 0:
                opt = tf.train.AdamOptimizer()
            else:
                opt = tf.train.AdamOptimizer(self._params['BASE_LEARNING_RATE'])

            # Variables for minibatching:
            self._zero_gradients =  [tv.assign(tf.zeros_like(tv)) for tv in self._accum_vars]
            self._accum_gradients = [self._accum_vars[i].assign_add(gv[0]) for
                                     i, gv in enumerate(opt.compute_gradients(self._loss))]
            self._apply_gradients = opt.apply_gradients(zip(self._accum_vars, tf.trainable_variables()),
                global_step = self._global_step)


        # Merge the summaries:
        self._merged_summary = tf.summary.merge_all()


    def apply_gradients(self,sess):

        return sess.run( [self._apply_gradients], feed_dict = {})


    def feed_dict(self, images, labels, weights=None):
        '''Build the feed dict

        Take input images, labels and (optionally) weights and match
        to the correct feed dict tensorrs

        Arguments:
            images {numpy.ndarray} -- Image array, [BATCH, L, W, F]
            labels {numpy.ndarray} -- Label array, [BATCH, L, W, F]

        Keyword Arguments:
            weights {numpy.ndarray} -- (Optional) input weights, same shape as labels (default: {None})

        Returns:
            [dict] -- Feed dictionary for a tf session run call

        Raises:
            IncompleteFeedDict -- If weights are requested in the configuration but not provided.
        '''
        fd = dict()
        fd.update({self._input_image : images})
        if labels is not None:
            fd.update({self._input_labels : labels})


        return fd

    def losses():
        pass

    def make_summary(self, sess, input_data, input_label, input_weight=None):
        fd = self.feed_dict(images  = input_data,
                            labels  = input_label,
                            weights = input_weight)
        return sess.run(self._merged_summary, feed_dict=fd)

    def zero_gradients(self, sess):
        sess.run(self._zero_gradients)

    def accum_gradients(self, sess, input_data, input_label, input_weight=None):

        feed_dict = self.feed_dict(images  = input_data,
                                   labels  = input_label,
                                   weights = input_weight)

        ops = [self._accum_gradients]
        doc = ['']
        # classification
        ops += [self._loss, self._total_accuracy]
        doc += ['loss', 'acc. all']

        return sess.run(ops, feed_dict = feed_dict ), doc


    def run_test(self,sess, input_data, input_label, input_weight=None):
        feed_dict = self.feed_dict(images   = input_data,
                                   labels  = input_label,
                                   weights = input_weight)

        ops = [self._loss, self._total_accuracy,]
        doc = ['loss', 'acc. all']

        return sess.run(ops, feed_dict = feed_dict ), doc

    def inference(self,sess,input_data,input_label=None):

        feed_dict = self.feed_dict(images=input_data, input_label=input_label)

        ops = [self._softmax]
        if input_label is not None:
          ops.append(self._accuracy_allpix)
          ops.append(self._accuracy_nonzero)

        return sess.run( ops, feed_dict = feed_dict )

    def global_step(self, sess):
        return sess.run(self._global_step)

    def _build_network(self, input_placeholder):

        x = input_placeholder
        # Initial convolution to get to the correct number of filters:
        x = tf.layers.conv3d(x, self._params['N_INITIAL_FILTERS'],
                             kernel_size=[5, 5, 5],
                             strides=[1, 1, 1],
                             padding='same',
                             use_bias=False,
                             trainable=self._params['TRAINING'],
                             name="Conv2DInitial",
                             reuse=None)

        # ReLU:
        x = tf.nn.relu(x)

        # Begin the process of residual blocks and downsampling:
        for i in xrange(self._params['NETWORK_DEPTH']):

            for j in xrange(self._params['RESIDUAL_BLOCKS_PER_LAYER']):
                x = residual_block(x, self._params['TRAINING'],
                                   batch_norm=self._params['BATCH_NORMALIZATION'],
                                   name="resblock_down_{0}_{1}".format(i, j))

            x = downsample_block(x, self._params['TRAINING'],
                                batch_norm=self._params['BATCH_NORMALIZATION'],
                                name="downsample_{0}".format(i))


        # At the bottom, do another residual block:
        for j in xrange(self._params['RESIDUAL_BLOCKS_PER_LAYER']):
            x = residual_block(x, self._params['TRAINING'],
                batch_norm=self._params['BATCH_NORMALIZATION'],
                name="deepest_block_{0}".format(j))

        # At this point, we ought to have a network that has the same shape as the initial input, but with more filters.
        # We can use a bottleneck to map it onto the right dimensions:
        x = tf.layers.conv3d(x,
                             self._params['NUM_LABELS'],
                             kernel_size=[1, 1, 1],
                             strides=[1, 1, 1],
                             padding='same',
                             activation=None,
                             use_bias=False,
                             trainable=self._params['TRAINING'],
                             name="BottleneckConv2D",)

        # Apply global average pooling to each filter to get the values for signal/bkg

        # For global average pooling, need to get the shape of the input:
        shape = (x.shape[1], x.shape[2], x.shape[3])

        x = tf.nn.pool(x,
                       window_shape=shape,
                       pooling_type="AVG",
                       padding="VALID",
                       dilation_rate=None,
                       strides=None,
                       name="GlobalAveragePool",
                       data_format=None)

        # Reshape to remove empty dimensions:
        x = tf.reshape(x, [tf.shape(x)[0], self._params['NUM_LABELS']],
                     name="global_pooling_reshape")

        # The final activation is softmax across the pixels.  It gets applied in the loss function
#         x = tf.nn.softmax(x)
        return x
