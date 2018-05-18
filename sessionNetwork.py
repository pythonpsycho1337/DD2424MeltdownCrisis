from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Parameters import *
import tensorflow as tf

class CNN_nonstatic:
    def __init__(self):

        #define placeholders
        self.batchX = tf.placeholder(tf.float32, [None, data_params.MAX_SENTENCE_SIZE, data_params.VOCAB_SIZE, 1])
        self.batchY = tf.placeholder(tf.int64, [None, 2])
        #self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        ######define network

        # define each group of filters
        layer_output = []
        for filter_height in network_params.FILTER_SIZES:
            conv = tf.layers.conv2d(
                inputs=  self.batchX, #########
                filters=network_params.NUM_FILTERS,
                kernel_size=(filter_height, data_params.VOCAB_SIZE),  # width of filter equals to the wordvec dimension
                padding="same",
                activation=tf.nn.relu)

            # max over time pooling
            pooling = tf.nn.max_pool(conv, ksize=[1, data_params.MAX_SENTENCE_SIZE - filter_height + 1, 1, 1],
                                     strides=[1, 1, 1, 1],
                                     padding='VALID',
                                     name="pool")

            layer_output.append(pooling)

        # concatenate the filter output
        concat_output = tf.concat(layer_output, 1)
        sum_filter_sizes = sum(network_params.FILTER_SIZES)
        reshape_output = tf.reshape(concat_output, [-1, sum_filter_sizes * data_params.VOCAB_SIZE * network_params.NUM_FILTERS])

        # Dense Layer for the dropout,
        dense = tf.layers.dense(inputs=reshape_output, units=network_params.DENSE_UNITS, activation=tf.nn.relu,
                                activity_regularizer=tf.contrib.layers.l2_regularizer(3.0))
        dropout = tf.layers.dropout(
            inputs=dense, rate=training_params.DROPOUT)  # dropout rate

        # Logits Layer
        logits = tf.layers.dense(inputs=dropout, units=2,activation=tf.nn.softmax)  # two classes (positive and negative)

        self.predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
             }

        #calculate cross entropy loss
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.batchY)
        self.loss = tf.reduce_mean(losses)   ##todo add regularization

        # Accuracy
        correct_predictions = tf.equal(self.predictions["classes"], tf.argmax(self.batchY, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")













