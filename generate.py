'''
generate.py - Generates the data files
'''

from preprocessing import word2vec_access_vector as w2v

print("")
data = w2v.data_word2vec_Twitter(False);
data = w2v.data_word2vec_MR();
