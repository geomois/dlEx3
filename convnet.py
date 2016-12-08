from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np



class ConvNet(object):
    """
   This class implements a convolutional neural network in TensorFlow.
   It incorporates a certain graph model to be trained and to be used
   in inference.
    """

    def __init__(self, n_classes = 10):
        """
        Constructor for an ConvNet object. Default values should be used as hints for
        the usage of each parameter.
        Args:
          n_classes: int, number of classes of the classification problem.
                          This number is required in order to specify the
                          output dimensions of the ConvNet.
        """
        self.n_classes = n_classes

    def inference(self, x):
        """
        Performs inference given an input tensor. This is the central portion
        of the network where we describe the computation graph. Here an input
        tensor undergoes a series of convolution, pooling and nonlinear operations
        as defined in this method. For the details of the model, please
        see assignment file.

        Here we recommend you to consider using variable and name scopes in order
        to make your graph more intelligible for later references in TensorBoard
        and so on. You can define a name scope for the whole model or for each
        operator group (e.g. conv+pool+relu) individually to group them by name.
        Variable scopes are essential components in TensorFlow for parameter sharing.
        Although the model(s) which are within the scope of this class do not require
        parameter sharing it is a good practice to use variable scope to encapsulate
        model.

        Args:
          x: 4D float Tensor of size [batch_size, input_height, input_width, input_channels]

        Returns:
          logits: 2D float Tensor of size [batch_size, self.n_classes]. Returns
                  the logits outputs (before softmax transformation) of the
                  network. These logits can then be used with loss and accuracy
                  to evaluate the model.
        """
        with tf.variable_scope('ConvNet'):
            ########################
            # PUT YOUR CODE HERE  #
            ########################
            # print ("x", x.get_shape())
            nnDict={}
            with tf.variable_scope("conv1"):
                # kernel=tf.get_variable("w",[5,5,3,64],regularizer=tf.contrib.layers.l2_regularizer(0.001),initializer=tf.contrib.layers.xavier_initializer())
                kernel=tf.get_variable("w",[5,5,3,64],initializer=tf.contrib.layers.xavier_initializer())
                bias=tf.get_variable("b",[64],initializer=tf.constant_initializer(0))
                self._variable_summaries(bias,tf.get_variable_scope().name+'/bias')
                self._variable_summaries(kernel,tf.get_variable_scope().name+'/weights')
                layer=tf.nn.conv2d(x,kernel, strides=[1,1,1,1], padding='SAME')
                pre_activation=tf.nn.bias_add(layer,bias)
                layer=tf.nn.relu(pre_activation,name='activation')
                nnDict['conv1']=layer
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
                nnDict['conv2']=layer
                layer=tf.nn.max_pool(layer,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
                tf.histogram_summary(tf.get_variable_scope().name+'/layer',layer)

            # reshape = tf.reshape(layer, [384, -1])
            # dim = reshape.get_shape()[1].value
            with tf.variable_scope("flatten"):
                flatten=tf.contrib.layers.flatten(layer,name='activation')
                nnDict['flatten']=flatten 
                tf.histogram_summary(tf.get_variable_scope().name+"/layer",layer)

            with tf.variable_scope("fc1"):
                # kernel=tf.get_variable("w",[flatten.get_shape()[1],384],regularizer=tf.contrib.layers.l2_regularizer(0.001),initializer=tf.contrib.layers.xavier_initializer())
                kernel=tf.get_variable("w",[flatten.get_shape()[1],384],initializer=tf.contrib.layers.xavier_initializer())
                bias=tf.get_variable("b",[384],initializer=tf.constant_initializer(0.1))
                self._variable_summaries(bias,tf.get_variable_scope().name+'/bias')
                self._variable_summaries(kernel,tf.get_variable_scope().name+'/weights')
                layer=tf.nn.relu(tf.add(tf.matmul(flatten,kernel),bias),name='activation')
                nnDict['fc1']=layer
                tf.histogram_summary(tf.get_variable_scope().name+'/layer',layer)

            with tf.variable_scope("fc2"):
                # kernel=tf.get_variable("w",[384,192],regularizer=tf.contrib.layers.l2_regularizer(0.001),initializer=tf.contrib.layers.xavier_initializer())
                kernel=tf.get_variable("w",[384,192],initializer=tf.contrib.layers.xavier_initializer())
                bias=tf.get_variable("b",[192],initializer=tf.constant_initializer(0.1))
                self._variable_summaries(bias,tf.get_variable_scope().name+'/bias')
                self._variable_summaries(kernel,tf.get_variable_scope().name+'/weights')
                layer=tf.nn.relu(tf.add(tf.matmul(layer,kernel),bias),name='activation')
                nnDict['fc2']=layer                
                tf.histogram_summary(tf.get_variable_scope().name+'/layer',layer)

            with tf.variable_scope("fc3"):
                # kernel=tf.get_variable("w",[192,self.n_classes],regularizer=tf.contrib.layers.l2_regularizer(0.001),initializer=tf.contrib.layers.xavier_initializer())
                kernel=tf.get_variable("w",[192,self.n_classes],initializer=tf.contrib.layers.xavier_initializer())
                bias=tf.get_variable("b",[self.n_classes],initializer=tf.constant_initializer(0.1))
                self._variable_summaries(bias,tf.get_variable_scope().name+'/bias')
                self._variable_summaries(kernel,tf.get_variable_scope().name+'/weights')
                layer=tf.add(tf.matmul(layer,kernel),bias)
                nnDict['out']=layer
                tf.histogram_summary(tf.get_variable_scope().name+'/softmax',layer)
            
            # logits=nnDict
            logits=layer
            ########################
            # END OF YOUR CODE    #
            ########################
        return logits


    def _variable_summaries(self,var, name):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.scalar_summary('stddev/' + name, stddev)
            tf.scalar_summary('max/' + name, tf.reduce_max(var))
            tf.scalar_summary('min/' + name, tf.reduce_min(var))
            tf.histogram_summary(name, var)


    def accuracy(self, logits, labels):
        """
        Calculate the prediction accuracy, i.e. the average correct predictions
        of the network.
        As in self.loss above, you can use tf.scalar_summary to save
        scalar summaries of accuracy for later use with the TensorBoard.

        Args:
          logits: 2D float Tensor of size [batch_size, self.n_classes].
                       The predictions returned through self.inference.
          labels: 2D int Tensor of size [batch_size, self.n_classes]
                     with one-hot encoding. Ground truth labels for
                     each observation in batch.

        Returns:
          accuracy: scalar float Tensor, the accuracy of predictions,
                    i.e. the average correct predictions over the whole batch.
        """
        ########################
        # PUT YOUR CODE HERE  #
        ########################
        with tf.name_scope("accuracy"):
            correct_predictions=tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
            accuracy=tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            tf.scalar_summary('accuracy',accuracy)
        ########################
        # END OF YOUR CODE    #
        ########################

        return accuracy

    def loss(self, logits, labels):
        """
        Calculates the multiclass cross-entropy loss from the logits predictions and
        the ground truth labels. The function will also add the regularization
        loss from network weights to the total loss that is return.
        In order to implement this function you should have a look at
        tf.nn.softmax_cross_entropy_with_logits.
        You can use tf.scalar_summary to save scalar summaries of
        cross-entropy loss, regularization loss, and full loss (both summed)
        for use with TensorBoard. This will be useful for compiling your report.

        Args:
          logits: 2D float Tensor of size [batch_size, self.n_classes].
                       The predictions returned through self.inference.
          labels: 2D int Tensor of size [batch_size, self.n_classes]
                       with one-hot encoding. Ground truth labels for each
                       observation in batch.

        Returns:
          loss: scalar float Tensor, full loss = cross_entropy + reg_loss
        """
        ########################
        # PUT YOUR CODE HERE  #
        ########################
        with tf.name_scope("loss"):
            # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy')
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy')
            loss = tf.reduce_mean(cross_entropy)
            loss=tf.add(loss, sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
            tf.scalar_summary('loss_regularized',loss)
        ########################
        # END OF YOUR CODE    #
        ########################

        return loss
