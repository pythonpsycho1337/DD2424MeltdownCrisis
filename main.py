from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from network import *
import tensorflow as tf

import preprocessing.word2vec_access_vector as wordvec
from train import *
from test import *

tf.logging.set_verbosity(tf.logging.INFO)


def main(execution_mode): #train or test

  # Load data (training and testing)
  data = wordvec.load_data('preprocessing/wordvectors/Google_Wordvec.npy', 'preprocessing/wordvectors/labels.npy')
  train_features = data[0]
  train_labels = data[1]
  test_features = data[2]
  test_labels = data[3]

  if (execution_mode == 'train'):
        train((train_features,train_labels), 'ckpt')
  elif (execution_mode == 'test'):
      test_network((test_features, test_labels), 'ckpt')


if __name__ == "__main__":
    #tf.app.run()
    main('train')


