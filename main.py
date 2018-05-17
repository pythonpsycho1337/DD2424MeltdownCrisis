'''
main.py - Runs the experiments for parameter exploration
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import preprocessing.word2vec_access_vector as wordvec
from train import *
from test import *
import os,re

#tf.logging.set_verbosity(tf.logging.INFO)

def main():
    # dataset takes values "MR" or "Twitter"
    dataset = "MR"
    if dataset=="MR":
        wordvecPath = "preprocessing/wordvectors_polarity/wordVecMR.npy"
        labelsPath = "preprocessing/wordvectors_polarity/labelsMR.npy"
    elif dataset=="Twitter":
        wordvecPath = "preprocessing/wordvectors_twitter/wordVecTwitter.npy"
        labelsPath = "preprocessing/wordvectors_twitter/labelsTwitter.npy"
    else:
        print("Could not find data for the dataset "+dataset)

    train_features, train_labels, val_features,val_labels, test_features, test_labels = wordvec.load_data(wordvecPath,labelsPath)

    print("succesfully loaded "+dataset+" dataset")

    trainingparams = {"TrainPercent":0.9,"LearningRateInit":0.1,"LearningDecay":0.95,"Dropout":0.5,"BatchSize":50,"Epochs":1,"Steps":1000}
    modelParams = {"FilterSizes":[3, 4, 5],"NumFilters":100,"l2Reg":3,"DenseUnits":100,"Rho":0.9}
    params = {"TrainingParams":trainingparams,"ModelParams":modelParams}
    modelDir = os.path.join("ckpt",paramsTodirName(params))
    train((train_features,train_labels),(val_features,val_labels), modelDir,params)
    test_network((test_features, test_labels), modelDir,params)

def paramsTodirName(params):
    #Creates a unique dirname based on the parameters
    regExPattern = '[A-Z]+'#Only keep capital letters in keys
    dir = ""
    for key in params['ModelParams'].keys():
        if key == "FilterSizes":
            dir += "".join(re.findall(regExPattern,key))+"".join([str(i) for i in params['ModelParams'][key]])
        else:
            dir += "".join(re.findall(regExPattern,key))+str(params['ModelParams'][key])

    for key in params['TrainingParams'].keys():
        dir += "".join(re.findall(regExPattern,key))+str(params['TrainingParams'][key])

    return dir

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    main()


