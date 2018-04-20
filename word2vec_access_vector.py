import gensim

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

vocab = model.vocab.keys()

i = model.vocab["for"]
vector = model.vectors[i]