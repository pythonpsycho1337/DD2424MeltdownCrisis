from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import word2vec_access_vector as wordvec
import Parameters as param

tf.logging.set_verbosity(tf.logging.INFO)


# Our application logic will be added here
def cnn_model_fn(features, labels, mode):
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

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)


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
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=adaptive_learning_rate, rho=param.RHO)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}

  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):

  # Load data (training and testing)
  data = wordvec.load_data('Google_Wordvec.npy', 'labels.npy')
  train_features = data[0]
  train_labels = data[1]
  test_features = data[2]
  test_labels = data[3]

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/model4")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=1)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x= {'x':train_features},
      y=train_labels,
      batch_size=param.BATCH_SIZE,
      num_epochs=param.EPOCHS,
      shuffle=True)

  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=param.STEPS,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x':test_features},
      y=test_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)

  print(eval_results)


if __name__ == "__main__":
    tf.app.run()

