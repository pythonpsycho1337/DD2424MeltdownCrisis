from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from network import *
import tensorflow as tf
from Parameters import *
from network import *


def train(trainingSet, validationSet,modelDir):
    BatchSize = 50
    numOfSteps = 5
    #Set model parameters
    modelParams = {"FilterSizes":[3, 4, 5],"NumFilters":100,"l2Reg":3,"DenseUnits":100,"Rho":0.9}

    # Create the Estimator
    run_config = tf.estimator.RunConfig(save_checkpoints_steps=STEPS).replace(
        session_config=tf.ConfigProto(log_device_placement=True))

    text_classifier = tf.estimator.Estimator(
        model_fn=cnn_basic, model_dir=modelDir, config=run_config, params=modelParams)

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
        batch_size=BatchSize,
        num_epochs=1,
        shuffle=True)

    text_classifier.train(
        input_fn=train_input_fn,
        steps=numOfSteps,
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
    np.save('validation_accuracy', accuracy_score)

    print("\nValidation Accuracy: {0:f}\n".format(accuracy_score))


    return text_classifier
#

