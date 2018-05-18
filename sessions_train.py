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
    training_loss_list = []
    validation_loss_list = []
    with sess.as_default():

        cnn_nonstatic = CNN_nonstatic(max_sentence_length, word2vec_dictionary) #define network graph

        ####define optimizer
        # adaptive learning rate - exponential decay
        global_step = tf.Variable(0, trainable=False)
        adaptive_learning_rate = tf.train.exponential_decay(training_params.LEARNING_RATE_INIT, global_step,
                                                            100, training_params.LEARNING_DECAY, staircase=True)
        #we minimize the sum of losses across all batches
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

        # performs one iteration across all batches (training or validation)
        # returns loss and accuracy
        def iterate_batches(shuffled_features, shuffled_labels):

            loss_epoch = 0
            # iterate through all batches
            train_size = shuffled_features.shape[0]
            num_batches = int((train_size - 1) / training_params.BATCH_SIZE) + 1
            for batch_num in range(num_batches):
                start_index = batch_num * training_params.BATCH_SIZE
                end_index = min((batch_num + 1) * training_params.BATCH_SIZE, train_size)

                X = shuffled_features[start_index:end_index]
                Y = shuffled_labels[start_index:end_index]

                ##define input tensor
                feed_dict = {
                    cnn_nonstatic.batchX: X,
                    cnn_nonstatic.batchY: Y
                }

                _, step, loss, accuracy = sess.run(
                    [train_op, global_step, cnn_nonstatic.loss, cnn_nonstatic.accuracy],
                    feed_dict)

                loss_epoch += loss  # compute total loss over all batches

            return (loss_epoch, accuracy)

        ####iterate thourgh epochs
        for epoch in range(training_params.EPOCHS):

            # shuffle training set
            (shuffled_features_train, shuffled_labels_train) = shuffle_set(training_set)

            # iterate through all training batches
            training_loss, training_accuracy = iterate_batches(shuffled_features_train, shuffled_labels_train)

            time_str = datetime.datetime.now().isoformat()
            print("time: ", time_str, "epoch: ", epoch, "loss: ", training_loss, "acc: ")


            training_loss_list.append(training_loss)

            #checkpointing mechanism
            current_step = tf.train.global_step(sess, global_step)
            if current_step % 10 == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

            #validate every a number of epochs
            if current_step % 10 == 0:

                # shuffle validation set
                (shuffled_features_val, shuffled_labels_val) = shuffle_set(validation_set)
                # iterate through all validation batches
                validation_loss, validation_accuracy = iterate_batches(shuffled_features_val, shuffled_labels_val)

                time_str = datetime.datetime.now().isoformat()
                print("validation loss: ", validation_loss, " validation accuracy: ", validation_accuracy)



        np.save("nonstatic_training_loss.npy", training_loss_list)
        np.save("nonstatic_validation_loss.npy", validation_loss_list)


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

    train(train_split, val_split, max_sentence_length, word2vec_array)

def shuffle_set(data): #(features, labels) shuffle train or validation set

    data_size = len(data[0])
    features = data[0]
    labels = data[1]
    shuffle_indx = np.random.permutation(np.arange(data_size))
    shuffled_features = features[shuffle_indx]
    shuffled_labels = labels[shuffle_indx]

    return (shuffled_features, shuffled_labels)



if __name__ == '__main__':
    tf.app.run()












