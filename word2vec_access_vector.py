import gensim
import data_preprocessing as pr
import numpy as np
from Parameters import *

#convert input data to word vector representation using word2vec trained Google model
def data_word2vec():
    data = pr.load_data_and_labels('datasets/rt-polaritydata/rt-polarity.neg', 'datasets/rt-polaritydata/rt-polarity.pos')
    sentences = data[0]

    # Load Google's pre-trained Word2Vec model.
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    vocab_size = model.vector_size  #vector size is 300 (num of words used from google vocabulary)


    #generate word vector representation of our data
    word_vectors = [] #list of all word2vec representations
    for i in range(len(sentences)):
        word_list = sentences[i].split(" ")
        mat = np.zeros((len(word_list), vocab_size)) #mat representation of each sentence
        for j in range(len(word_list)):
            #check if word not found in dictionary
            if (word_list[j] in model.vocab.keys()):
                idx = int(model.vocab[word_list[j]].index)
                mat[j,:] = model.vectors[idx]
            else:
                print('not valid word!')
                mat[j,:] = -1 #not valid word vector

        mat = mat[~np.all(mat == -1, axis=1)]  #delete non valid vectors
        word_vectors.append(mat)

    np.save('Google_Wordvec', word_vectors) #save vector representations
    np.save('labels', data[1])  #save labels to file

    return word_vectors

#load files with vectors and labels and split into training and testing
def load_data(filename_vec, filename_labels):

    vectors = np.load(filename_vec)
    labels = np.load(filename_labels)

    #transform labels to 1 column from 2 columns (labels is 0 or 1)
    labels_vec = np.zeros((labels.shape[0]))
    for i in range(labels.shape[0]):
        if (labels[i,0] != 0):
            labels_vec[i] = 0
        else:
            labels_vec[i] = 1

    labels = labels_vec.astype(int)

    #find maximum sentence size
    sizes = []
    for i in range(len(vectors)):
        sizes.append(vectors[i].shape[0])
    max_dim = max(sizes)

    #zero padding each matrix to the maximum height
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

    num_elements = vectors[0].shape[0] * vectors[0].shape[1]
    for i in range(vectors.size):
       vectors[i] = np.reshape(vectors[i],(num_elements))

    flatten_vectors = np.vstack(vectors)
    flatten_vectors = np.float32(flatten_vectors)

    #split into training and testing
    train_size = int(TRAIN_PERCENT * data_size)
    train_vectors = flatten_vectors[0:train_size]
    train_labels = labels[0:train_size]
    test_vectors = flatten_vectors[train_size:]
    test_labels = labels[train_size:]


    return [train_vectors, train_labels, test_vectors, test_labels]








