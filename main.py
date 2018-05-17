from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import preprocessing.word2vec_access_vector as wordvec
import sys
from train import *
from test import *

#tf.logging.set_verbosity(tf.logging.INFO)

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
    modelParams = {"FilterSizes":[3, 4, 5],"NumFilters":100,"l2Reg":3,"DenseUnits":100,"Rho":0.9}
    trainingparams ={}
    params = {"TrainingParams":trainingparams,"ModelParams":modelParams}
    train((train_features,train_labels),(val_features,val_labels), 'ckpt',params)
    test_network((test_features, test_labels), 'ckpt',params)


if __name__ == "__main__":
    main()


