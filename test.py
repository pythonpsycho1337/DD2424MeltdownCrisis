from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from network import *
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)

#predict classes using loaded model
#todo fix error
def test_network(testdata, dir):

    test_features = testdata[0]
    test_labels = testdata[1]

    #load saved model
    classifier = tf.estimator.Estimator(model_fn=cnn_basic, model_dir=dir)

    #create input tensor
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': test_features},
        y= test_labels,
        num_epochs=1,
        shuffle=False)

    prediction_results = classifier.predict(input_fn=test_input_fn)

    for x, each in enumerate(prediction_results):
        print(each)


