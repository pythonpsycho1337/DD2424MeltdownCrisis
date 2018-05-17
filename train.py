from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os, sys
import pickle

from Parameters import training_params
from network import cnn_basic

tf.reset_default_graph()


def train(train_set, validation_set, dir):

    #trainable_list = get_trainable_list(dir)

    # Create the Estimator
    run_config = tf.estimator.RunConfig(save_checkpoints_steps=training_params.STEPS).replace(
        session_config=tf.ConfigProto(log_device_placement=True))

    text_classifier = tf.estimator.Estimator(
        model_fn=cnn_basic, model_dir=dir, config=run_config, params=4)


    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}  #,"input_features":"input"}
    #tensors_to_log = {"input_features":"input_variable"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1)

    #create input tensor for training
    train_features = train_set[0]
    train_labels = train_set[1]
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_features},
        y=train_labels,
        batch_size=training_params.BATCH_SIZE,
        num_epochs=training_params.EPOCHS,
        shuffle=True)


    text_classifier.train(
        input_fn=train_input_fn,
        steps=training_params.STEPS,
        hooks=[logging_hook])



    # Evaluate the model on validation set and print results
    validation_features = validation_set[0]
    validation_labels = validation_set[1]
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': validation_features},
        y=validation_labels,
        num_epochs=1,
        shuffle=False)

    # Evaluate accuracy.
    accuracy_score = text_classifier.evaluate(input_fn=eval_input_fn)["accuracy"]
    np.save('validation_accuracy', accuracy_score)

    print("\nValidation Accuracy: {0:f}\n".format(accuracy_score))

    return accuracy_score



