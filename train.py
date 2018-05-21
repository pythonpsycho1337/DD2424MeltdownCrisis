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

def train(trainingSet, validationSet, modelDir,params):

    #set up estimator configurations
    run_config = tf.estimator.RunConfig(save_checkpoints_steps=100, save_summary_steps=10).replace(
        session_config=tf.ConfigProto(log_device_placement=True))

    # Create the Estimator
    text_classifier = tf.estimator.Estimator(
        model_fn=network.cnn_basic, model_dir=modelDir, config=run_config, params=params)

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    #tensors_to_log = {"probabilities": "softmax_tensor"}
    #logging_hook = tf.train.LoggingTensorHook(
     #   tensors=tensors_to_log, every_n_iter=1)


    train_features = trainingSet[0]
    train_labels = trainingSet[1]
    # Returns input function that would feed dict of numpy arrays into the training function.
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_features},
        y=train_labels,
        batch_size=params["TrainingParams"]["BatchSize"],
        num_epochs=params["TrainingParams"]["Epochs"],
        shuffle=True)

    # passes the train_input_fn function to the train function.
    # That will split the data into batches and feed them in the form they are supposed to be fed into the train function
    #also passes the steps of the training and what we want to be logged as info in the standard output
    text_classifier.train(
        input_fn=train_input_fn,
        steps=None) #will be defined by input function
       # hooks=[logging_hook])

    # Evaluate the model on validation set and print results
    validation_features = validationSet[0]
    validation_labels = validationSet[1]
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': validation_features},
        y=validation_labels,
        num_epochs=1,
        shuffle=False)

    # Evaluate accuracy.
    valAcc = text_classifier.evaluate(input_fn=eval_input_fn)["accuracy"]
    np.save(os.path.join(modelDir,'validation_accuracy'), valAcc)

    tf.summary.scalar("validationAcc",valAcc)

    print("\nValidation Accuracy: {0:f}\n".format(valAcc))


    ##==== test network ====#
#
    #test_features = testSet[0]
    #test_labels = testSet[1]
    ## Create input tensor
    #test_input_fn = tf.estimator.inputs.numpy_input_fn(
    #   x={'x': test_features},
    #   num_epochs=1,
    #   shuffle=False)
    #predictions = list(text_classifier.predict(input_fn=test_input_fn))
    #np.save(os.path.join(modelDir, 'predictions'), predictions)
    #print("prediction is done!")
    ## Measure the accuracy on the test set by calling the EVALUATE function
    #eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #   x={'x': test_features},
    #   y=test_labels,
    #   batch_size=params["TrainingParams"]["BatchSize"],
    #   num_epochs=1,
    #   shuffle=False)
    #testAcc = text_classifier.evaluate(input_fn=eval_input_fn)["accuracy"]
    #np.save(os.path.join(modelDir, 'testing_accuracy'), testAcc)
    #tf.summary.scalar("testAcc", testAcc)
#
    #print("\nAccuracy on test set: {0:f}\n".format(testAcc))




    return valAcc

