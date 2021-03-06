'''
main.py - Runs the experiments for parameter exploration
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle,os,re,time

import preprocessing.word2vec_access_vector as wordvec
from train import *
from test import *

def main():
    dataset = "Twitter"#Takes values "MR" or "Twitter"
    if dataset=="MR":
        wordvecPath = "preprocessing/wordvectors_polarity/wordVecMR.npy"
        labelsPath = "preprocessing/wordvectors_polarity/labelsMR.npy"
    elif dataset=="Twitter":
        wordvecPath = "preprocessing/wordvectors_twitter/wordVecTwitter.npy"
        labelsPath = "preprocessing/wordvectors_twitter/labelsTwitter.npy"
    else:
        print("Could not find data for the dataset "+dataset)

    train_features, train_labels, val_features,val_labels, test_features, test_labels = wordvec.load_data(wordvecPath,labelsPath)
    print("=== dataset sizes ===")
    print("train_set: ", train_labels.size, " dev_set: ", val_labels.size, " test_set: ", test_labels.size)
    print("succesfully loaded "+dataset+" dataset")

    log = []
    for i in range(1):#For loop is customized for different parameter settings for each experiment
        trainingParams = {"TrainPercent":0.7,"LearningRateInit":0.01,"LearningDecay":0.95,"Dropout":0.5,"BatchSize":50,"Epochs":20}
        modelParams = {"FilterSizes":[6,7,8],"NumFilters":100, "DenseUnits":100,"Rho":0.9}
        params = {"TrainingParams":trainingParams,"ModelParams":modelParams}
        modelDir = os.path.join("ckpt",dataset,"BEST", paramsTodirName(params))

        start = time.time()
        valAcc = train((train_features,train_labels),(val_features,val_labels), modelDir,params)
        testAcc = test_network((test_features, test_labels), modelDir,params)

        log.append({"Time":time.time()-start,"valAcc":valAcc,"testAcc":testAcc,"trainingParams":trainingParams,"modelParams":modelParams})
        f = open("Log.pkl", "wb")
        pickle.dump(log,f)
        f.close()#Close the file to force the update

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


