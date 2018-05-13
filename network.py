from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import Parameters as param

tf.logging.set_verbosity(tf.logging.INFO)

#basic CNN network with one channel
def cnn_basic(features, labels, mode, params):

  """Model function for CNN."""  #input image size (46,300) - one channel
  # Input Layer
  max_sentence_size = 46  #max sentence size
  vocab_size = 300  #wordvec dimensions
  input_layer = tf.reshape(features['x'], [-1, max_sentence_size, vocab_size, 1]) #-1 corresponds to the batch (dynamicallly calculated), 1 corresponds to the channel used

  #define each group of filters
  layer_output = []
  for filter_height in param.FILTER_SIZES:

      conv = tf.layers.conv2d(
          inputs=input_layer,
          filters=param.NUM_FILTERS,
          kernel_size = (filter_height, vocab_size), #width of filter equals to the wordvec dimension
          padding="same",
          activation=tf.nn.relu)

      # max over time pooling
      pooling = tf.nn.max_pool(conv, ksize=[1, max_sentence_size - filter_height + 1, 1, 1],
                               strides=[1, 1, 1, 1],
                               padding='VALID',
                               name="pool")

      layer_output.append(pooling)

  # concatenate the filter output
  concat_output = tf.concat(layer_output, 1)
  sum_filter_sizes = sum(param.FILTER_SIZES)
  reshape_output = tf.reshape(concat_output, [-1, sum_filter_sizes * vocab_size * param.NUM_FILTERS])

  # Dense Layer for the dropout,
  dense = tf.layers.dense(inputs=reshape_output, units=param.DENSE_UNITS, activation=tf.nn.relu,
                          activity_regularizer=tf.contrib.layers.l2_regularizer(3.0))
  dropout = tf.layers.dropout(
      inputs=dense, rate=param.DROPOUT, training=mode == tf.estimator.ModeKeys.TRAIN)  # dropout rate

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=2, activation=tf.nn.softmax)  # two classes (positive and negative)

  # prediction MODE
  predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


  # adaptive learning rate - exponential decay
  global_step = tf.Variable(0, trainable=False)
  adaptive_learning_rate = tf.train.exponential_decay(param.LEARNING_RATE_INIT, global_step,
                                                        100, param.LEARNING_DECAY, staircase=True)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:

      loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

      optimizer = tf.train.AdadeltaOptimizer(learning_rate=adaptive_learning_rate, rho=param.RHO)
      train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

      # Configure the Training Op (for EVAL mode)
  else: # mode == tf.estimator.ModeKeys.EVAL:

      loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
      eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}

      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


