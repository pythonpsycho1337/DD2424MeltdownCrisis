from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Parameters import *
import tensorflow as tf
from sessionNetwork import CNN_nonstatic
from preprocessing.data_preprocessing import load_data_and_labels
from preprocessing.word2vec_access_vector import splitData

import time
import argparse
import os
import shutil
import numpy as np
import datetime
import preprocessing.word2vec_access_vector as wordvec


def train(training_set, validation_set, max_sentence_length, word2vec_dictionary):


    session_conf = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
    sess = tf.Session(config=session_conf)
    with sess.as_default():

        cnn_nonstatic = CNN_nonstatic(max_sentence_length, word2vec_dictionary) #define network graph

        ####define optimizer
        # adaptive learning rate - exponential decay
        global_step = tf.Variable(0, trainable=False)
        adaptive_learning_rate = tf.train.exponential_decay(training_params.LEARNING_RATE_INIT, global_step,
                                                            100, training_params.LEARNING_DECAY, staircase=True)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=adaptive_learning_rate, rho=network_params.RHO)
        train_op = optimizer.minimize(
            loss=cnn_nonstatic.loss,
            global_step=tf.train.get_global_step())

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))


        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        # Initialize all variables before training
        sess.run(tf.global_variables_initializer())

        ######iterate in epochs
        for i in range(training_params.EPOCHS):

            #shuffle training set
            (shuffled_features, shuffled_labels) = shuffle_set(training_set)

            #iterate through all training batches
            train_size = shuffled_features.shape[0]
            num_batches = int((train_size  - 1)/training_params.BATCH_SIZE) + 1
            for batch_num in range(num_batches):
                start_index = batch_num * training_params.BATCH_SIZE
                end_index = min((batch_num + 1) * training_params.BATCH_SIZE, train_size )

                X_train = shuffled_features[start_index:end_index]
                Y_train = shuffled_labels[start_index:end_index]


                ##training step
                feed_dict = {
                    cnn_nonstatic.batchX: X_train,
                    cnn_nonstatic.batchY: Y_train
                }
                _, step, loss, accuracy = sess.run(
                    [train_op, global_step, cnn_nonstatic.loss, cnn_nonstatic.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("--iteration i: ", i)
                print("time: ", time_str, "step: ",step, "loss: ",loss, "acc: ", accuracy)

                # checkpointing mechanism
                current_step = tf.train.global_step(sess, global_step)
                if current_step % 10 == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

                #validate every a number of epochs
               # if current_step % 10 == 0:

                    # shuffle validation set
                    (shuffled_features_val, shuffled_labels_val) = shuffle_set(validation_set)
                    #todo: iterate through all validation batches for one epoch and go through the network
                    #todo: create a function that iterates through all batches and returns a list with loss and accuracy





def main(argv=None):
    dataset = "MR"
    if dataset == "MR":
        dataPosPath = "preprocessing/wordvectors_polarity/rt-polarity.pos"
        dataNegPath = "preprocessing/wordvectors_polarity/rt-polarity.neg"
    else:
        print("Could not find data for the dataset " + dataset)

    data, max_sentence_length, unique_word_dictionary = wordvec.load_word_dataset(dataPosPath, dataNegPath)
    train_split = data[0:2]
    val_split = data[2:4]
    test_split = data[4:6]


    word2vec_dictionary = wordvec.load_obj("/preprocessing/wordvectors_polarity/", "word2vecDict")
    word2vec_array = np.zeros((18758,300))
    for k, v in word2vec_dictionary.items():
        word2vec_array[k] = v

    # evi's code
    # (vec_matrix, label_matrix) = prepare_data()
    # split = splitData(vec_matrix, label_matrix, 0.1, 0.1)
    # train_split = split[0:2]
    # val_split = split[2:4]
    # test_split = split[4:6]

    train(train_split, val_split, max_sentence_length, word2vec_array)

def shuffle_set(data): #(features, labels) shuffle train or validation set

    data_size = len(data[0])
    features = data[0]
    labels = data[1]
    shuffle_indx = np.random.permutation(np.arange(data_size))
    shuffled_features = features[shuffle_indx]
    shuffled_labels = labels[shuffle_indx]

    return (shuffled_features, shuffled_labels)



#features format: 4d array
#labels format: 2d array
def prepare_data():

    vectors = np.load("preprocessing/wordvectors_polarity/wordVecMR.npy")
    original_data = load_data_and_labels('datasets/rt-polaritydata/rt-polarity.pos',
                                         'datasets/rt-polaritydata/rt-polarity.neg')
    labels = original_data[1]

    # zero pad features
    # find maximum sentence size
    sizes = []
    for i in range(len(vectors)):
        sizes.append(vectors[i].shape[0])
    max_dim = max(sizes)

    # zero padding each matrix to the maximum height
    # each element of vectors is a matrix (max_sentence_length, vocabulary_length)
    vector_dim = vectors[0].shape[1]
    vec_matrix = np.zeros((len(vectors), data_params.MAX_SENTENCE_SIZE, data_params.VOCAB_SIZE))  ###3d array!!!
    label_matrix = np.zeros((len(vectors), 2))
    for i in range(len(vectors)):
        res = max_dim - vectors[i].shape[0]
        z = np.zeros((res, vector_dim))
        vec_matrix[i] = np.vstack((vectors[i], z))
        label_matrix[i] = labels[i].T

    vec_matrix = vec_matrix.reshape((len(vectors), data_params.MAX_SENTENCE_SIZE, data_params.VOCAB_SIZE, 1))

    return vec_matrix, label_matrix




if __name__ == '__main__':
    tf.app.run()












