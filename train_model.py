from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
import numpy as np
import cifar10_utils
from convnet import *
from sklearn.manifold import TSNE

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'

DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
LOG_DIR_DEFAULT = './logs/cifar10'
CHECKPOINT_DIR_DEFAULT = './checkpoints'

OPTIMIZER_DICT = {'sgd': tf.train.GradientDescentOptimizer, # Gradient Descent
                  'adadelta': tf.train.AdadeltaOptimizer, # Adadelta
                  'adagrad': tf.train.AdagradOptimizer, # Adagrad
                  'adam': tf.train.AdamOptimizer, # Adam
                  'rmsprop': tf.train.RMSPropOptimizer # RMSprop
                  }

def train_step(loss):
    """
    Defines the ops to conduct an optimization step. You can set a learning
    rate scheduler or pick your favorite optimizer here. This set of operations
    should be applicable to both ConvNet() and Siamese() objects.

    Args:
        loss: scalar float Tensor, full loss = cross_entropy + reg_loss

    Returns:
        train_op: Ops for optimization.
    """
    ########################
    # PUT YOUR CODE HERE  #
    ########################
    optimizer=OPTIMIZER_DICT['adam'](learning_rate=FLAGS.learning_rate)
    train_op=optimizer.minimize(loss)
    ########################
    # END OF YOUR CODE    #
    ########################

    return train_op

def train():
    """
    Performs training and evaluation of ConvNet model.

    First define your graph using class ConvNet and its methods. Then define
    necessary operations such as trainer (train_step in this case), savers
    and summarizers. Finally, initialize your model within a tf.Session and
    do the training.

    ---------------------------
    How to evaluate your model:
    ---------------------------
    Evaluation on test set should be conducted over full batch, i.e. 10k images,
    while it is alright to do it over minibatch for train set.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every print_freq iterations
    - on test set every eval_freq iterations

    ------------------------
    Additional requirements:
    ------------------------
    Also you are supposed to take snapshots of your model state (i.e. graph,
    weights and etc.) every checkpoint_freq iterations. For this, you should
    study TensorFlow's tf.train.Saver class. For more information, please
    checkout:
    [https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html]
    """

    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    
    x_test, y_test = cifar10.test.images, cifar10.test.labels
    x_pl=tf.placeholder(tf.float32,shape=(None,x_test.shape[1],x_test.shape[2],x_test.shape[3]))
    y_pl=tf.placeholder(tf.float32,shape=(None,y_test.shape[1]))
    
    convNet=ConvNet()
    prediction=convNet.inference(x_pl)
    pred=prediction['out']
    loss=convNet.loss(pred,y_pl)
    accuracy=convNet.accuracy(pred,y_pl)
    train_op=train_step(loss)
    saver=tf.train.Saver()

    #Trying and approach with collections
    # tf.add_to_collection('nn',prediction['flatten'])
    # tf.add_to_collection('nn',prediction['fc1'])
    # tf.add_to_collection('nn',prediction['fc2'])
    # tf.add_to_collection('nn',pred)
    # tf.add_to_collection('nn',accuracy)
    # tf.add_to_collection('fc2',prediction['fc2'])
    # tf.add_to_collection('fc2',prediction['fc1'])

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        train_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/train',sess.graph)
        test_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/test',sess.graph)

        for epoch in xrange(FLAGS.max_steps +1):
            batch_x, batch_y = cifar10.train.next_batch(FLAGS.batch_size)
            _,out,acc=sess.run([train_op,loss,accuracy], feed_dict={x_pl: batch_x,y_pl: batch_y})
            if epoch % FLAGS.print_freq == 0:
                # train_writer.add_summary(merged_sum,epoch)
                # train_writer.flush()
                print ("Epoch:", '%05d' % (epoch), "loss=","{:.4f}".format(out),"accuracy=","{:.4f}".format(acc))
            if epoch % FLAGS.eval_freq ==0 and epoch>0:
                # batch_x, batch_y = cifar10.test.next_batch(FLAGS.batch_size*5)
                avgLoss=0
                avgAcc=0
                count=0
                step=1000
                for i in xrange(0,x_test.shape[0],step):
                    batch_x=x_test[i:i+step]
                    batch_y=y_test[i:i+step]
                    loss,acc=sess.run([loss,accuracy], feed_dict={x_pl: batch_x,y_pl: batch_y})
                    avgAcc=avgAcc+acc
                    avgLoss=avgLoss+loss
                    count=count+1
                # out,acc=sess.run([loss,accuracy], feed_dict={x_pl: x_test,y_pl:y_test})
                # test_writer.afdd_summary(merged_sum,epoch)
                # test_writer.flush()
                print ("Test set:","accuracy=","{:.4f}".format(avgAcc/count))
            if epoch % FLAGS.checkpoint_freq==0 and epoch>0:
                saver.save(sess,FLAGS.checkpoint_dir+'/linear'+str(epoch)+'.ckpt')

    ########################
    # END OF YOUR CODE    #
    ########################

# def fetch_test_batch():


def train_siamese():
    """
    Performs training and evaluation of Siamese model.

    First define your graph using class Siamese and its methods. Then define
    necessary operations such as trainer (train_step in this case), savers
    and summarizers. Finally, initialize your model within a tf.Session and
    do the training.

    ---------------------------
    How to evaluate your model:
    ---------------------------
    On train set, it is fine to monitor loss over minibatches. On the other
    hand, in order to evaluate on test set you will need to create a fixed
    validation set using the data sampling function you implement for siamese
    architecture. What you need to do is to iterate over all minibatches in
    the validation set and calculate the average loss over all minibatches.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every print_freq iterations
    - on test set every eval_freq iterations

    ------------------------
    Additional requirements:
    ------------------------
    Also you are supposed to take snapshots of your model state (i.e. graph,
    weights and etc.) every checkpoint_freq iterations. For this, you should
    study TensorFlow's tf.train.Saver class. For more information, please
    checkout:
    [https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html]
    """

    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    ########################


def feature_extraction():
    """
    This method restores a TensorFlow checkpoint file (.ckpt) and rebuilds inference
    model with restored parameters. From then on you can basically use that model in
    any way you want, for instance, feature extraction, finetuning or as a submodule
    of a larger architecture. However, this method should extract features from a
    specified layer and store them in data files such as '.h5', '.npy'/'.npz'
    depending on your preference. You will use those files later in the assignment.

    Args:
        [optional]
    Returns:
        None
    """

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    x_test, y_test = cifar10.test.images, cifar10.test.labels
    x_pl=tf.placeholder(tf.float32,shape=(None,x_test.shape[1],x_test.shape[2],x_test.shape[3]))
    y_pl=tf.placeholder(tf.float32,shape=(None,y_test.shape[1]))
    
    print("Building the model")
    convNet=ConvNet()
    pred=convNet.inference(x_pl)
    loss=convNet.loss(pred,y_pl)
    accuracy=convNet.accuracy(pred,y_pl)
    
    #Taking intermediate layers
    flatten = tf.get_default_graph().get_tensor_by_name("ConvNet/flatten/activation:0")
    fc1 = tf.get_default_graph().get_tensor_by_name("ConvNet/fc1/activation:0")
    fc2 = tf.get_default_graph().get_tensor_by_name("ConvNet/fc2/activation:0")

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
        graph  = tf.get_default_graph()
        check = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir+"/")
        sess.run(tf.initialize_all_variables())

        saver.restore(sess,check.model_checkpoint_path)

        acc, flattenOut,fc1Out,fc2Out = sess.run([accuracy,flatten,fc1, fc2],feed_dict={x_pl: x_test,y_pl:y_test})

    #     avgLoss=0
    #     avgAcc=0
    #     count=0
    #     step=1000
    #     #Feed forward whole test_set in batches and keep the average (gpu limitations) 
    #     for i in xrange(0,x_test.shape[0],step):
    #         batch_x=x_test[i:i+step]
    #         batch_y=y_test[i:i+step]
    #         acc=sess.run([accuracy], feed_dict={x_pl: batch_x,y_pl: batch_y})
    #         avgAcc=avgAcc+acc
    #         avgLoss=avgLoss+loss
    #         count=count+1
    #     print ("Test set:","accuracy=","{:.4f}".format(avgAcc/count))
        np.save("./features/flatten.npy", flattenOut)
        np.save("./features/fc1.npy", fc1Out)
        np.save("./features/fc2.npy", fc2Out)        

    ########################
    # END OF YOUR CODE    #
    ########################

def initialize_folders():
    """
    Initializes all folders in FLAGS variable.
    """

    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)

    if not tf.gfile.Exists(FLAGS.data_dir):
        tf.gfile.MakeDirs(FLAGS.data_dir)

    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def main(_):
    print_flags()

    initialize_folders()
    if FLAGS.is_train:
        if FLAGS.train_model == 'linear':
            train()
        elif FLAGS.train_model == 'siamese':
            train_siamese()
        else:
            raise ValueError("--train_model argument can be linear or siamese")
    else:
        feature_extraction()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
    parser.add_argument('--print_freq', type = int, default = PRINT_FREQ_DEFAULT,
                      help='Frequency of evaluation on the train set')
    parser.add_argument('--eval_freq', type = int, default = EVAL_FREQ_DEFAULT,
                      help='Frequency of evaluation on the test set')
    parser.add_argument('--checkpoint_freq', type = int, default = CHECKPOINT_FREQ_DEFAULT,
                      help='Frequency with which the model state is saved.')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
    parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')
    parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
                      help='Checkpoint directory')
    parser.add_argument('--is_train', type = str, default = True,
                      help='Training or feature extraction')
    parser.add_argument('--train_model', type = str, default = 'linear',
                      help='Type of model. Possible options: linear and siamese')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
