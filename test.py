from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import network

#tf.logging.set_verbosity(tf.logging.INFO)

def test_network( testSet, modelDir,params):
    #Params is a dictionary containing two dictionaries refered to by the keys 'modelParams' and 'trainingParams'
    classifier = tf.estimator.Estimator(model_fn=network.cnn_basic, model_dir=modelDir, params=params)

    test_features = testSet[0]
    test_labels = testSet[1]

    #Create input tensor
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': test_features},
        num_epochs=1,
        shuffle=False)

    predictions = list(classifier.predict(input_fn=test_input_fn))
    np.save('predictions', predictions)

    #Measure the accuracy on the test set by calling the EVALUATE function
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': test_features},
        y=test_labels,
        num_epochs=1,
        shuffle=False)
    accuracy_score = classifier.evaluate(input_fn=eval_input_fn)["accuracy"]
    np.save('testing_accuracy', accuracy_score)

    print("\nAccuracy on test set: {0:f}\n".format(accuracy_score))
