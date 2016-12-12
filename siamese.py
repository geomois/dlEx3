from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np



class Siamese(object):
    """
    This class implements a siamese convolutional neural network in
    TensorFlow. Term siamese is used to refer to architectures which
    incorporate two branches of convolutional networks parametrized
    identically (i.e. weights are shared). These graphs accept two
    input tensors and a label in general.
    """

    def inference(self, x, reuse = False):
        """
        Defines the model used for inference. Output of this model is fed to the
        objective (or loss) function defined for the task.

        Here we recommend you to consider using variable and name scopes in order
        to make your graph more intelligible for later references in TensorBoard
        and so on. You can define a name scope for the whole model or for each
        operator group (e.g. conv+pool+relu) individually to group them by name.
        Variable scopes are essential components in TensorFlow for parameter sharing.
        You can use the variable scope to activate/deactivate 'variable reuse'.

        Args:
           x: 4D float Tensor of size [batch_size, input_height, input_width, input_channels]
           reuse: Python bool to switch reusing on/off.

        Returns:
           l2_out: L2-normalized output tensor of shape [batch_size, 192]

        Hint: Parameter reuse indicates whether the inference graph should use
        parameter sharing or not. You can study how to implement parameter sharing
        in TensorFlow from the following sources:

        https://www.tensorflow.org/versions/r0.11/how_tos/variable_scope/index.html
        """
        with tf.variable_scope('ConvNet') as conv_scope:
            ########################
            # PUT YOUR CODE HERE  #
            ########################
            # tf.cond(reuse,tf.get_variable_scope().reuse_variables(),)
            if (reuse):
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("conv1"):
                # kernel=tf.get_variable("w",[5,5,3,64],regularizer=tf.contrib.layers.l2_regularizer(0.001),initializer=tf.contrib.layers.xavier_initializer())
                kernel=tf.get_variable("w",[5,5,3,64],initializer=tf.contrib.layers.xavier_initializer())
                bias=tf.get_variable("b",[64],initializer=tf.constant_initializer(0))
                self._variable_summaries(bias,tf.get_variable_scope().name+'/bias')
                self._variable_summaries(kernel,tf.get_variable_scope().name+'/weights')
                layer=tf.nn.conv2d(x,kernel, strides=[1,1,1,1], padding='SAME')
                pre_activation=tf.nn.bias_add(layer,bias)
                layer=tf.nn.relu(pre_activation,name='activation')
                layer=tf.nn.max_pool(layer,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
                # conv1=tf.nn.max_pool(conv1,ksize=[3,3],strides=[1,2,2,1],padding='SAME')
                tf.histogram_summary(tf.get_variable_scope().name+'/layer',layer)

            with tf.variable_scope("conv2"):
                # kernel=tf.get_variable("w",[5,5,64,64],regularizer=tf.contrib.layers.l2_regularizer(0.001),initializer=tf.contrib.layers.xavier_initializer())
                kernel=tf.get_variable("w",[5,5,64,64],initializer=tf.contrib.layers.xavier_initializer())
                bias=tf.get_variable("b",[64],initializer=tf.constant_initializer(0))
                self._variable_summaries(bias,tf.get_variable_scope().name+'/bias')
                self._variable_summaries(kernel,tf.get_variable_scope().name+'/weights')
                layer=tf.nn.conv2d(layer,kernel, strides=[1,1,1,1], padding='SAME')
                pre_activation=tf.nn.bias_add(layer,bias)
                layer=tf.nn.relu(pre_activation,name='activation')
                layer=tf.nn.max_pool(layer,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
                tf.histogram_summary(tf.get_variable_scope().name+'/layer',layer)

            # reshape = tf.reshape(layer, [384, -1])
            # dim = reshape.get_shape()[1].value
            with tf.variable_scope("flatten"):
                flatten=tf.contrib.layers.flatten(layer)
                tf.histogram_summary(tf.get_variable_scope().name+"/layer",layer,name='activation')

            with tf.variable_scope("fc1"):
                # kernel=tf.get_variable("w",[flatten.get_shape()[1],384],regularizer=tf.contrib.layers.l2_regularizer(0.001),initializer=tf.contrib.layers.xavier_initializer())
                kernel=tf.get_variable("w",[flatten.get_shape()[1],384],initializer=tf.contrib.layers.xavier_initializer())
                bias=tf.get_variable("b",[384],initializer=tf.constant_initializer(0.1))
                self._variable_summaries(bias,tf.get_variable_scope().name+'/bias')
                self._variable_summaries(kernel,tf.get_variable_scope().name+'/weights')
                layer=tf.nn.relu(tf.add(tf.matmul(flatten,kernel),bias),name='activation')
                tf.histogram_summary(tf.get_variable_scope().name+'/layer',layer)

            with tf.variable_scope("fc2"):
                # kernel=tf.get_variable("w",[384,192],regularizer=tf.contrib.layers.l2_regularizer(0.001),initializer=tf.contrib.layers.xavier_initializer())
                kernel=tf.get_variable("w",[384,192],initializer=tf.contrib.layers.xavier_initializer())
                bias=tf.get_variable("b",[192],initializer=tf.constant_initializer(0.1))
                self._variable_summaries(bias,tf.get_variable_scope().name+'/bias')
                self._variable_summaries(kernel,tf.get_variable_scope().name+'/weights')
                layer=tf.nn.relu(tf.add(tf.matmul(layer,kernel),bias),name='activation')
                tf.histogram_summary(tf.get_variable_scope().name+'/layer',layer)
                
            with tf.variable_scope("l2_norm"):
                layer=tf.nn.l2_normalize(layer,0, epsilon=1e-12, name="outNorm")#να τσεκάρω dim
                # tf.nn.l2_normalize(layer,1 , epsilon=1e-12, name=scope.name)#να τσεκάρω dim

            l2_out=layer
            ########################
            # END OF YOUR CODE    #
            ########################

        return l2_out

    def loss(self, channel_1, channel_2, label, margin):
        """
        Defines the contrastive loss. This loss ties the outputs of
        the branches to compute the following:

               L =  Y * d^2 + (1-Y) * max(margin - d^2, 0)

               where d is the L2 distance between the given
               input pair s.t. d = ||x_1 - x_2||_2 and Y is
               label associated with the pair of input tensors.
               Y is 1 if the inputs belong to the same class in
               CIFAR10 and is 0 otherwise.

               For more information please see:
               http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

        Args:
            channel_1: output of first channel (i.e. branch_1),
                              tensor of size [batch_size, 192]
            channel_2: output of second channel (i.e. branch_2),
                              tensor of size [batch_size, 192]
            label: Tensor of shape [batch_size]
            margin: Margin of the contrastive loss

        Returns:
            loss: scalar float Tensor
        """
        ########################
        # PUT YOUR CODE HERE  #
        ########################
        with tf.name_scope("contrastive_loss"):
            d = tf.reduce_sum(tf.square(tf.sub(channel_1,channel_2), 1)#keep_dims=True
            d_sqrt = tf.sqrt(d)
            loss = label * tf.square(tf.maximum(0., margin - d_sqrt)) + (1 - label) * d
        ########################
        # END OF YOUR CODE    #
        ########################

        return loss
