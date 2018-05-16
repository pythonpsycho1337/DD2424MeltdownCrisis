from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from network import *
from Parameters import *
import tensorflow as tf
import numpy as np
from network import *


tf.logging.set_verbosity(tf.logging.INFO)

#predict classes using loaded model
def test_network( testdata, dir):

    classifier = tf.estimator.Estimator(model_fn=cnn_basic, model_dir=dir)

    test_features = testdata[0]
    test_labels = testdata[1]

    #create input tensor
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': test_features},
        num_epochs=1,
        shuffle=False)

    predictions = list(classifier.predict(input_fn=test_input_fn))
    np.save('predictions', predictions)

    #measure the accuracy on the test set calling the EVALUATE function
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': test_features},
        y=test_labels,
        num_epochs=1,
        shuffle=False)
    accuracy_score = classifier.evaluate(input_fn=eval_input_fn)["accuracy"]
    np.save('testing_accuracy', accuracy_score)

    print("\nAccuracy on test set: {0:f}\n".format(accuracy_score))