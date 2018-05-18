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

def get_trainable_list(modeldir):

    #load latest checkpoint file
    #latest_ckpt_file = estimator_obj.latest_checkpoint()
    checkpoint = tf.train.get_checkpoint_state(modeldir) #, latest_filename=latest_ckpt_file)
    chosen_graph = 'ckpt/model.ckpt-21.meta'
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(chosen_graph)
        saver.restore(sess, checkpoint.model_checkpoint_path)

        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        print('TRAINABLE VARIABLES!!!!')
        print(trainable_vars)


    return trainable_vars

def train(train_set, validation_set, dir):

    trainable_list = get_trainable_list(dir)
    trainable_dct_list = []
    for var in trainable_list:
        var_scope = var.name.split('/')[0]
        var_name = var.name.split('/')[1].split(':')[0]
        d = {'name':var_name, 'scope':var_scope, 'dtype':var.dtype, 'shape':var.shape}
        trainable_dct_list.append(d)

    # Create the Estimator
    run_config = tf.estimator.RunConfig(save_checkpoints_steps=training_params.STEPS).replace(
        session_config=tf.ConfigProto(log_device_placement=True))

    text_classifier = tf.estimator.Estimator(
        model_fn=cnn_basic, model_dir=dir, config=run_config, params=trainable_dct_list)


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

    print('*****')
    #print(text_classifier.latest_checkpoint())
    #print(text_classifier.get_variable_names())
    #get_trainable_list(text_classifier, dir)

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



