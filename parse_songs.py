import os
import re
import numpy as np
import pickle
from numpy import genfromtxt
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.datasets import dump_svmlight_file


# this is probably not the best way to encode
# a one-hot label from the song path

def encode_label(genres,song_path):
    label = None
    for idx, genre in enumerate(genres):
        if re.search(genre,song_path):
            label = idx
            # print(genre)
            print(song_path)
            # print(label)
            break
    return label

def vectorize_song_feature(filepath):
    song_features = genfromtxt(filepath, delimiter=",")

def create_data_set(data_set_path,keyword=None,lower_limit=None,upper_limit=None,verbose=True,categorical=True):

    training_vector = []
    labels = []

    for root, dirs, files in os.walk(data_set_path, topdown=False):
        genres = [_dir for _dir in dirs]
        # print(genres)
    # print("genres",genres)
    # print(data_set_path)
    for root, dirs, files in os.walk(data_set_path, topdown=False):
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
                training_vector.append(song_features)

                labels.append(encode_label(genres,song_path))
                # print(encode_label(genres,song_path))

    if categorical:
        # print(labels)
        labels = np_utils.to_categorical(labels)
    maxlen = np.max([len(vector) for vector in training_vector])
    return training_vector,labels,maxlen

def build_vectors(keyword="",data_label="",lower_limit=None,upper_limit=None):
    # training
    training_vector,labels,maxlen_training = create_data_set(data_set_path = "dataset/train",keyword=keyword,lower_limit=lower_limit,upper_limit=upper_limit)

    # validation
    evaluation_training_vector,evaluation_labels,maxlen_evaluation = create_data_set(data_set_path = "dataset/test",keyword=keyword,lower_limit=lower_limit,upper_limit=upper_limit)

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

if __name__=="__main__":

    keyword = "spectral-contrast_peaks"

    training_vector,labels,maxlen_training = create_data_set(data_set_path = "dataset/train",keyword=keyword,lower_limit=1)
    evaluation_training_vector,evaluation_labels,maxlen_evaluation = create_data_set(data_set_path = "dataset/test",keyword=keyword,lower_limit=1)

    with(open("maxlen","w")) as f:
        f.write(":".join([keyword,str(np.max([maxlen_evaluation,maxlen_training]))]))

    keyword = "mfcc_coefficients"

    training_vector,labels,maxlen_training = create_data_set(data_set_path = "dataset/train",keyword=keyword,lower_limit=1)
    evaluation_training_vector,evaluation_labels,maxlen_evaluation = create_data_set(data_set_path = "dataset/test",keyword=keyword,lower_limit=1)

    with(open("maxlen","a")) as f:
        f.write(":".join([keyword,str(np.max([maxlen_evaluation,maxlen_training]))]))

    # build_vectors(keyword="spectral-contrast_peaks",lower_limit=1)
    # build_vectors(keyword="mfcc_coefficients",lower_limit=1)
    # build_vectors(keyword="tempotracker_tempo",upper_limit=-1)
