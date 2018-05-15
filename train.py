from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from network import *
import tensorflow as tf
from Parameters import *
from network import *


def train(traindata, dir):

    # Create the Estimator
    run_config = tf.estimator.RunConfig(save_checkpoints_steps=STEPS).replace(
        session_config=tf.ConfigProto(log_device_placement=True))

    text_classifier = tf.estimator.Estimator(
        model_fn=cnn_basic, model_dir=dir, config=run_config)

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor","input_features":"input"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1)


    #create input tensor
    train_features = traindata[0]
    train_labels = traindata[1]
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_features},
        y=train_labels,
        batch_size=param.BATCH_SIZE,
        num_epochs=param.EPOCHS,
        shuffle=True)

    text_classifier.train(
        input_fn=train_input_fn,
        steps=param.STEPS,
        hooks=[logging_hook])


   # # Evaluate the model and print results  #todo create validation set!
   # eval_input_fn = tf.estimator.inputs.numpy_input_fn(
   #     x={'x': validation_features},
   #     y=validation_labels,
   #     num_epochs=1,
   #     shuffle=False)
##
   # #eval_results = text_classifier.evaluate(input_fn=eval_input_fn)["accuracy"]
##
   # # Evaluate accuracy.
   # accuracy_score = text_classifier.evaluate(input_fn=eval_input_fn)["accuracy"]
#
   # print("\nValidation Accuracy: {0:f}\n".format(accuracy_score))
#
    return text_classifier
#

