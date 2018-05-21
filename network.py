'''
network.py - Defines the network structure
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

#tf.logging.set_verbosity(tf.logging.INFO)

#basic CNN network with one channel
def cnn_basic(features, labels, mode, params):
  modelParams = params["ModelParams"]
  DROPOUT = params["TrainingParams"]["Dropout"]
  LEARNING_RATE_INIT = params["TrainingParams"]["LearningRateInit"]
  LEARNING_DECAY = params["TrainingParams"]["LearningDecay"]

  """Model function for CNN."""
  batch_shape = features['x'].get_shape()
  batch_width = batch_shape[1].value

  # Input Layer
  word_vector_size = 300  # wordvec dimensions
  max_sentence_size =int(batch_width/ word_vector_size)#batch width = max_sentence_size * word_vector_size
  print('max sentence size: ', max_sentence_size)

  input_layer = tf.reshape(features['x'], [-1, max_sentence_size, word_vector_size, 1])

  #define each group of filters
  layer_output = []
  for filter_height in modelParams['FilterSizes']:

      conv = tf.layers.conv2d(
          inputs=input_layer,
          filters=modelParams['NumFilters'],
          kernel_size = (filter_height, word_vector_size), #width of filter equals to the wordvec dimension
          padding="same",
          activation=tf.nn.tanh)

      # max over time pooling
      pooling = tf.nn.max_pool(conv,
                               ksize=[1, max_sentence_size - filter_height + 1, 1, 1],
                               strides=[1, 1, 1, 1],
                               padding='VALID',
                               name="pool")

      layer_output.append(pooling)

  # concatenate the filter output
  concat_output = tf.concat(layer_output, 1)
  sum_filter_sizes = sum(modelParams["FilterSizes"])
  reshape_output = tf.reshape(concat_output, [-1, sum_filter_sizes * word_vector_size * modelParams['NumFilters']])

  # Dense Layer for the dropout,
  dense = tf.layers.dense(inputs=reshape_output, units=modelParams['DenseUnits'], activation=tf.nn.relu)
                          #activity_regularizer=tf.contrib.layers.l2_regularizer(0.5))
  dropout = tf.layers.dropout(
      inputs=dense, rate=DROPOUT, training=mode == tf.estimator.ModeKeys.TRAIN)  # dropout rate

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
      print("prediction mode")
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


  # adaptive learning rate - exponential decay
  global_step = tf.Variable(0, trainable=False)
  adaptive_learning_rate = tf.train.exponential_decay(LEARNING_RATE_INIT, global_step,
                                                        100, LEARNING_DECAY, staircase=True)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  tf.summary.scalar("Finalloss", loss)
  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
      #print("Loss:"+str(loss))
      optimizer = tf.train.MomentumOptimizer(learning_rate=adaptive_learning_rate, momentum=0.9)
      train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Configure the Training Op (for EVAL mode)
  elif mode == tf.estimator.ModeKeys.EVAL:
      eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}

      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
  else:
        print("WARNING: No mode specified for training")

