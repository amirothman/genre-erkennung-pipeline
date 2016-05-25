import vamp
import librosa
from split_30_seconds_mono import iterate_audio
import numpy as np
import re
import os
import pickle

# this is probably not the best way to encode label

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

def create_vectors(audio_format,folder_path,plugin_key):

    for root, dirs, files in os.walk(folder_path, topdown=False):
        genres = [_dir for _dir in dirs]
        # print(genres)

    X = []
    y = []

    for song_path in iterate_audio(audio_format,folder_path):
        data,rate = librosa.load(song_path)
        mfcc = vamp.collect(data,rate,plugin_key)
        X.append([np.array(el) for el in mfcc["matrix"][1]])
        y.append(encode_label(genres,song_path))

    return (X,y)

def save_vectors(plugin_key,
                  audio_format,
                  data_folder,
                  save_folder_path="pickled_vectors",
                  data_label="",
                  pickle_vectors=True,
                  ):

    # data_folder is where the folders "train" and "test"
    # need to be located

    format_plugin_key = re.sub(r":","_",plugin_key)

    # training_vector

    training_vector,training_label = create_vectors(audio_format,
                                                    "/".join([data_folder,"train"]),
                                                    plugin_key)

    pickle.dump(training_vector,open("pickled_vectors/{1}{0}_training_vector.pickle".format(format_plugin_key,data_label),"wb"))
    pickle.dump(training_label,open("pickled_vectors/{1}{0}_label.pickle".format(format_plugin_key,data_label),"wb"))
    # test_vector

    test_vector,test_label = create_vectors(audio_format,
                                                    "/".join([data_folder,"test"]),
                                                    plugin_key)

    pickle.dump(test_vector,open("pickled_vectors/{1}{0}_evaluation_vector.pickle".format(format_plugin_key,data_label),"wb"))
    pickle.dump(test_label,open("pickled_vectors/{1}{0}_evaluation_label.pickle".format(format_plugin_key,data_label),"wb"))

    # print(training_vector)
    maxlen = np.max([len(vector) for vector in training_vector])
    maxlen_test = np.max([len(vector) for vector in test_vector])
    vector_maxlen = np.max([maxlen,maxlen_test])

    return(training_vector,training_label,test_training_vector,test_label,vector_maxlen)

if __name__=="__main__":
    folder_path = "dataset/boom"
    plugin_key = "qm-vamp-plugins:qm-mfcc"
    audio_format = "mp3"

    X,y,X_val,y_val,maxlen = save_vectors(plugin_key,audio_format,folder_path,data_label="dummy")
