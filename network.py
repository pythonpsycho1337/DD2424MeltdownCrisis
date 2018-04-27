from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import word2vec_access_vector as wordvec

tf.logging.set_verbosity(tf.logging.INFO)


# Our application logic will be added here
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""  #input image size (46,300) - one channel
  # Input Layer
  dim_x = 46  #max sentence size
  dim_y = 300  #wordvec dimensions
  input_layer = tf.reshape(features, [-1, dim_x, dim_y, 1]) #-1 corresponds to the batch (dynamicallly calculated), 1 corresponds to the channel used

  # Convolutional Layer - one for each filter size
  filter_sizes = [(3, dim_y), (4, dim_y), (5, dim_y)]
  pooled_output = []

  #define each group of filters
  layer_output = []
  for size in filter_sizes:

      conv = tf.layers.conv2d(
          inputs=input_layer,
          filters=100,
          kernel_size=size, #width of filter equals to the wordvec dimension
          padding="same",
          activation=tf.nn.relu)

      # max over time pooling
      pooling = tf.nn.max_pool(conv, ksize=[1,  46-size[0]+1, 1, 1],
                                      strides=[1,1,1,1],
                                      padding='VALID',
                                      name="pool")

      layer_output.append(pooling)


  #concatenate the filter output
  concat_output = tf.concat(layer_output, 1)  #size=(100,3+4+5,300,100)
  reshape_output = tf.reshape(concat_output, [-1,12*300*100])


 # Dense Layer for the dropout,
  dense = tf.layers.dense(inputs=reshape_output, units=100, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)  #dropout rate

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=2, activation=tf.nn.softmax)   #two classes (positive and negative)

  b=logits.eval()
  a=logits

  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):

  # Load data (training and testing)
  data = wordvec.load_data('Google_Wordvec.npy', 'labels.npy')
  train_features = data[0]
  train_labels = data[1]
  test_features = data[2]
  test_labels = data[3]

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/model1")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x= train_features,
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={test_features},
      y=test_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)

  print(eval_results)



if __name__ == "__main__":
 tf.app.run()

