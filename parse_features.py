#!/usr/local/bin/python3
import os
import re
import numpy as np
import pickle
import sys
from numpy import genfromtxt
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.datasets import dump_svmlight_file


# this is probably not the best way to encode
# a one-hot label from the song path

def encode_label(genres,song_path,verbose=True):
    label = None
    for idx, genre in enumerate(genres):
        if re.search(genre,song_path):
            label = idx
            # print(genre)
            if verbose:
                print(song_path)
            # print(label)
            break
    return label

def vectorize_song_feature(filepath):
    song_features = genfromtxt(filepath, delimiter=",")

def create_dataset(dataset_path, keyword=None, lower_limit=None, upper_limit=None, verbose=True, categorical=True):
    training_vector = []
    labels = []

    for root, dirs, files in os.walk(dataset_path, topdown=False):
        genres = [_dir for _dir in dirs]
        # print(genres)
    print("genres",genres)
    # print(dataset_path)
    for root, dirs, files in os.walk(dataset_path, topdown=False):
        for name in files:
            if re.search("{0}.csv".format(keyword),name):
                song_path = (os.path.join(root,name))
                if verbose:
                    print(song_path)

                song_features = genfromtxt(song_path, delimiter=",")

                if len(song_features.shape) is 2:
                    song_features = np.array([ _line[lower_limit:upper_limit] for _line in song_features])
                elif len(song_features.shape) is 1:
                    song_features = np.array([song_features[lower_limit:upper_limit]])

                if song_features.shape[1] == 0:
                    print("song_features.shape = (x,0)")
                else:
                    # print(song_features.shape)
                    training_vector.append(song_features)
                    labels.append(encode_label(genres,song_path,verbose=verbose))
                # print(encode_label(genres,song_path))

    if categorical:
        # print(labels)
        labels = np_utils.to_categorical(labels)
    maxlen = np.max([len(vector) for vector in training_vector])
    return training_vector,labels,maxlen

def get_vector(dataset_path, keyword=None, lower_limit=None, upper_limit=None, verbose=True, categorical=True):
    training_vector = []
    for root, dirs, files in os.walk(dataset_path, topdown=False):
        for name in files:
            if re.search("{0}.csv".format(keyword),name):
                song_path = (os.path.join(root,name))
                if verbose:
                    print(song_path)

                song_features = genfromtxt(song_path, delimiter=",")

                if len(song_features.shape) is 2:
                    song_features = np.array([ _line[lower_limit:upper_limit] for _line in song_features])
                elif len(song_features.shape) is 1:
                    song_features = np.array([song_features[lower_limit:upper_limit]])

                if song_features.shape[1] == 0:
                    print("song_features.shape = (x,0)")
                else:
                    # print(song_features.shape)
                    training_vector.append(song_features)
    return training_vector

def build_vectors(keyword="",data_label="",lower_limit=None,upper_limit=None,folder_path="dataset"):

    # training
    training_vector,labels,maxlen_training = create_dataset(dataset_path = folder_path+"/train",keyword=keyword,lower_limit=lower_limit,upper_limit=upper_limit)

    # validation
    evaluation_training_vector,evaluation_labels,maxlen_evaluation = create_dataset(dataset_path = "{0}/test".format(folder_path),keyword=keyword,lower_limit=lower_limit,upper_limit=upper_limit)

    # # X_training
    training_vector = sequence.pad_sequences(training_vector, maxlen=np.max([maxlen_training,maxlen_evaluation]),dtype='float32')
    pickle.dump(training_vector,open("pickled_vectors/{1}{0}_training_vector.pickle".format(keyword,data_label),"wb"))
    #
    # # y
    #
    pickle.dump(labels,open("pickled_vectors/{1}{0}_label.pickle".format(keyword,data_label),"wb"))
    #
    #
    # # evaluation
    evaluation_training_vector = sequence.pad_sequences(evaluation_training_vector, maxlen=np.max([maxlen_training,maxlen_evaluation]),dtype='float32')
    pickle.dump(evaluation_training_vector,open("pickled_vectors/{1}{0}_evaluation_training_vector.pickle".format(keyword,data_label),"wb"))
    #
    # # evaluation
    pickle.dump(evaluation_labels,open("pickled_vectors/{1}{0}_evaluation_label.pickle".format(keyword,data_label),"wb"))
    with(open("maxlen_{0}".format(keyword),"w")) as _f:
        _f.write(str(np.max([maxlen_training,maxlen_evaluation])))

if __name__=="__main__":
    if len(sys.argv) < 2:
        print("missing parameter for dataset path")
    else:
        path=sys.argv[1]
        if not os.path.exists("pickled_vectors"):
            os.makedirs("pickled_vectors")
        build_vectors(folder_path=path,keyword="spectral-contrast_peaks")
        build_vectors(folder_path=path,keyword="mfcc_coefficients")
        # build_vectors(keyword="tempotracker_tempo",upper_limit=-1)
        #create_dataset("dataset/my_dataset",keyword="spectral-contrast_peaks",lower_limit=1)
        #create_dataset("dataset/my_dataset",keyword="mfcc_coefficients",lower_limit=1)
