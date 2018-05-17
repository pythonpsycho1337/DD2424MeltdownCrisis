'''
word2vec_access_vector.py - Generate word vectors from sentences and load data
'''

import numpy as np
import os,gensim

from preprocessing import data_preprocessing as pr
#convert input data to word vector representation using word2vec trained Google model
def data_word2vec_MR():
    path1 = os.path.join(os.getcwd(),"datasets","rt-polaritydata","rt-polarity.neg")
    path2 = os.path.join(os.getcwd(), "datasets", "rt-polaritydata", "rt-polarity.pos")
    [sentences,labels] = pr.load_data_and_labels(path1,path2)
    wordVecs = data_word2vec(sentences)
    saveWordVecsAndLabels(wordVecs,labels,'MR')
    return wordVecs

def data_word2vec_SST():
    #Not done
    data = pr.load_SST('datasets/stanfordSentimentTreebank/datasetSentences.txt',
                        'datasets/stanfordSentimentTreebank/datasetSplit.txt',
                       'datasets/stanfordSentimentTreebank/sentiment_labels.txt')
    return data_word2vec(data)

def data_word2vec_Twitter(includeNeutral):
    #Done
    path = os.path.join(os.getcwd(), "datasets", "Twitter2017-4A-English", "TwitterData.txt")
    [sentences, labels] = pr.load_twitter(path,includeNeutral)
    wordVecs = data_word2vec(sentences)
    saveWordVecsAndLabels(wordVecs, labels, 'Twitter')
    return wordVecs

def data_word2vec(sentences):
    #sentences: input list of form list

    # Load Google's pre-trained Word2Vec model.
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    vocab_size = model.vector_size  #vector size is 300 (num of words used from google vocabulary)

    #generate word vector representation of our data
    word_vectors = [] #list of all word2vec representations
    for i in range(len(sentences)):
        print("%.2f%%" % (i/len(sentences)*100))
        word_list = sentences[i].split(" ")
        mat = np.zeros((len(word_list), vocab_size)) #mat representation of each sentence
        for j in range(len(word_list)):
            #check if word not found in dictionary
            if (word_list[j] in model.vocab.keys()):
                idx = int(model.vocab[word_list[j]].index)
                mat[j,:] = model.vectors[idx]
            else:
                #print('not valid word!')
                mat[j,:] = -1 #not valid word vector

        mat = mat[~np.all(mat == -1, axis=1)]  #delete non valid vectors
        word_vectors.append(mat)

    return word_vectors

def load_data(filename_vec, filename_labels):
    # load files with vectors and labels and split into training and testing
    vectors = np.load(filename_vec)
    labels = np.load(filename_labels)

    #find maximum sentence size
    sizes = []
    for i in range(len(vectors)):
        sizes.append(vectors[i].shape[0])
    max_dim = max(sizes)

    #zero padding each matrix to the maximum height
    #each element of vectors is a matrix (max_sentence_length, vocabulary_length)
    vector_dim = vectors[0].shape[1]
    for i in range(len(vectors)):
        res = max_dim - vectors[i].shape[0]
        z = np.zeros((res, vector_dim))
        vectors[i] = np.vstack((vectors[i], z))

    #data shuffling
    data_size = len(labels)
    indx = np.random.randint(data_size, size=data_size)
    vectors = vectors[indx]
    labels = labels[indx]

    #transform list of arrays into a matrix (number of reviews, max_sentence_length * vocabulary_length)
    num_elements = vectors[0].shape[0] * vectors[0].shape[1]
    flatten_vectors = np.empty((vectors.size,  num_elements))

    for i in range(vectors.size):
        flatten_vectors[i] = np.reshape(vectors[i],(num_elements))

    flatten_vectors = np.float32(flatten_vectors)

    return splitData(flatten_vectors,labels,0.1,0.1)

def saveWordVecsAndLabels(wordVecs,labels,name):
    np.save('wordVec'+name, wordVecs)  # save wordVecs to file
    np.save('labels'+name, labels)  # save labels to file

def splitData(data,labels,testPercent,validationPercent):
    # Split data into training,validation and test sets
    # data is the flattened vectors
    examples = data.shape[0]
    testLim = int((1-testPercent) * examples)
    testData = data[testLim:]
    testLabels = labels[testLim:]
    data = data[:testLim]
    labels = labels[:testLim]

    examples = data.shape[0]
    validationLim = int((1 - validationPercent) * examples)
    validationData = data[validationLim:]
    validationLabels = labels[validationLim:]
    trainingData = data[:validationLim]
    trainingLabels = labels[:validationLim]

    return trainingData, trainingLabels, validationData, validationLabels, testData, testLabels


if __name__ == "__main__":
    data = data_word2vec_MR()
    #data_word2vec_SST()
    data_word2vec_Twitter()






