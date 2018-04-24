import gensim
import data_preprossesing as pr
import numpy as np

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

    np.save('Google_Wordvec', word_vectors)

    return word_vectors

wordvec = data_word2vec()
