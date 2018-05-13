from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import preprocessing.word2vec_access_vector as wordvec
import sys
from train import *
from test import *

tf.logging.set_verbosity(tf.logging.INFO)

def main():
    dataset = sys.argv[1]
    if dataset=="MR":
        wordvecPath = "preprocessing/wordvectors_polarity/wordVecMR.npy"
        labelsPath = "preprocessing/wordvectors_polarity/labelsMR.npy"
    elif dataset=="Twitter":
        wordvecPath = "preprocessing/wordvectors_twitter/wordVecTwitter.npy"
        labelsPath = "preprocessing/wordvectors_twitter/labelsTwitter.npy"
    else:
        print("Could not find data for the dataset "+dataset)
    #dataset takes values "MR" or "Twitter"
    data = wordvec.load_data(wordvecPath,labelsPath)
    train_features = data[0][:100,:]
    train_labels = data[1][:100]
    test_features = data[2][:100,:]
    test_labels = data[3][:100]

    print("succesfully loaded "+dataset+" dataset")
    train((train_features,train_labels), 'ckpt')
    test_network((test_features, test_labels), 'ckpt')


if __name__ == "__main__":
    main("MR")


