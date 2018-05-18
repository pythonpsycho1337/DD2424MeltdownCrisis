'''
train.py - File for training the network
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os

import network

def train(trainingSet, validationSet,modelDir,params):
    # Create the Estimator
    run_config = tf.estimator.RunConfig(save_checkpoints_steps=10).replace(
        session_config=tf.ConfigProto(log_device_placement=True))

    text_classifier = tf.estimator.Estimator(
        model_fn=network.cnn_basic, model_dir=modelDir, config=run_config, params=params)

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1)


    #create input tensor
    train_features = trainingSet[0]
    train_labels = trainingSet[1]
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_features},
        y=train_labels,
        batch_size=params["TrainingParams"]["BatchSize"],
        num_epochs=params["TrainingParams"]["Epochs"],
        shuffle=True)

    text_classifier.train(
        input_fn=train_input_fn,
        steps=params["TrainingParams"]["Steps"],
        hooks=[logging_hook])

    # Evaluate the model on validation set and print results
    validation_features = validationSet[0]
    validation_labels = validationSet[1]
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': validation_features},
        y=validation_labels,
        num_epochs=1,
        shuffle=False)

    # Evaluate accuracy.
    accuracy_score = text_classifier.evaluate(input_fn=eval_input_fn)["accuracy"]
    np.save(os.path.join(modelDir,'validation_accuracy'), accuracy_score)

    tf.summary.scalar("validationAcc",accuracy_score)

    print("\nValidation Accuracy: {0:f}\n".format(accuracy_score))

    return text_classifier

