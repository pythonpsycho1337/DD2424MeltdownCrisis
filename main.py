from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import preprocessing.word2vec_access_vector as wordvec
from train import *
from test import *

tf.logging.set_verbosity(tf.logging.INFO)

def main():
  # Load data (training and testing)
  data = wordvec.load_data('preprocessing/wordvectors_polarity/Google_Wordvec.npy', 'preprocessing/wordvectors_polarity/labels.npy')
  train_features = data[0][:100,:]
  train_labels = data[1][:100]
  test_features = data[2][:100,:]
  test_labels = data[3][:100]


  train((train_features,train_labels), 'ckpt')
  test_network((test_features, test_labels), 'ckpt')


if __name__ == "__main__":
    #tf.app.run()
    main()


