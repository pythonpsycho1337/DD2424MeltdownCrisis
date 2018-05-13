from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from network import *
import tensorflow as tf

import preprocessing.word2vec_access_vector as wordvec
from train import *
from test import *

tf.logging.set_verbosity(tf.logging.INFO)


def main():

  # Load data (training and testing)
  data = wordvec.load_data('preprocessing/wordvectors_polarity/Google_Wordvec.npy', 'preprocessing/wordvectors_polarity/labels.npy')
  trainingData = data[0]
  trainingLabels = data[1]
  validationData = data[2]
  validationLabels = data[3]
  testData = data[4]
  testLabels = data[5]

  train((trainingData,trainingLabels), (validationData,validationLabels), 'ckpt')
  test_network((testData,  testLabels), 'ckpt')


if __name__ == "__main__":
    #tf.app.run()
    main()



