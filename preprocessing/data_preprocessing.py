'''
data_preprocessing.py - Preprocess different datasets
'''

import numpy as np
import re

# ---------- preprossesing of rt-polaritydata -----------#
def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]

    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y_onehot = np.concatenate([positive_labels, negative_labels], 0)

    return [x_text, y_onehot]

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Inspired by https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub('"', '', string)

    return string.strip().lower()

def load_twitter(fileName,includeNeutral=True):
    #Load the twitter data
    twitterFile = open(fileName,"r")
    sentences = []
    labels = []
    translationDict = {'negative':0,'positive':1,'neutral':2}
    lines = twitterFile.readlines()
    for i in range(0,len(lines)):
        # Remove id
        index = lines[i].find("\t")
        lines[i] = lines[i][index+1:]

        # Extract label
        labelIndex = lines[i].find("\t")
        label = translationDict[lines[i][:labelIndex]]
        if (includeNeutral == False and label == 2):
            continue
        labels.append(label)

        # Extract sentence
        lines[i] = lines[i][labelIndex+1:]
        lines[i] = clean_str(lines[i])
        sentences.append(lines[i])

    labels = np.asarray(labels)
    return [sentences,labels]