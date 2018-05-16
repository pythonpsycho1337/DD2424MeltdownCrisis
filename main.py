from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import preprocessing.word2vec_access_vector as wordvec
from train import train
import sys
#from testing_env import hyperparameter_optimization

def main():
    dataset = "MR"
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
    train_features = data[0]
    train_labels = data[1]
    val_features = data[2]
    val_labels = data[3]
    test_features = data[4]
    test_labels = data[5]

    print("succesfully loaded "+dataset+" dataset")
   # hyperparameter_optimization(data, 'filter_size', {'start':0, 'end':10}, 'ckpt')
    val_accuracy = train((train_features,train_labels), (val_features,val_labels), 'ckpt')
    #test_network((test_features, test_labels), 'ckpt')


if __name__ == "__main__":
    main()


