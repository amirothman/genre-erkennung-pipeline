from parse_songs import build_vectors
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import model_from_json, Sequential
import json
import pickle

# parse features

def parse_features():
    build_vectors(keyword="spectral-contrast_peaks",
                  lower_limit=1,
                  folder_path="dataset/gztan_split_10sec",
                  data_label="10_sec_split_gztan_")

    build_vectors(keyword="mfcc_coefficients",
                  lower_limit=1,
                  folder_path="dataset/gztan_split_10sec",
                  data_label="10_sec_split_gztan_")

def load_test_data():
    y_test = pickle.load(open("pickled_vectors/10_sec_split_gztanmfcc_coefficients_evaluation_label.pickle","rb"))
    X_test_2 = pickle.load(open("pickled_vectors/10_sec_split_gztanmfcc_coefficients_evaluation_training_vector.pickle","rb"))
    X_test_3 = pickle.load(open("pickled_vectors/10_sec_split_gztanspectral-contrast_peaks_evaluation_training_vector.pickle","rb"))
    return (X_test_2,X_test_3,y_test)

if __name__=="__main__":

    # parse_features()
    build_vectors(keyword="tempotracker_tempo",
                  lower_limit=1,
                  folder_path="dataset/gztan_split_10sec",
                  data_label="10_sec_split_gztan_")
    # X_2,X_3,y = load_test_data()
    #
    # json_string = json.load(open("model_architecture/gztan_merged_model_architecture.json","r"))
    # model = model_from_json(json_string)
    # model.load_weights("model_weights/gztan_merged_model_weights.hdf5")
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy']
    #               )
    #
    # genres = ['metal', 'hiphop', 'pop', 'disco', 'classical', 'reggae', 'jazz', 'rock', 'country', 'blues']
    #
    # y_by_genres = [[] for i in range(10)]
    # limits = []
    # for idx,el_y in enumerate(y):
    #     list_el_y = list(el_y)
    #     genre_idx = list_el_y.index(1)
    #     y_by_genres[genre_idx].append(np.array(el_y))
    #
    # for idx,el_y in enumerate(y_by_genres):
    #     length = len(el_y)
    #     if idx == 0:
    #         limits.append((0,length-1))
    #     else:
    #         previous_limit = limits[idx-1][1]+1
    #         limits.append((previous_limit,previous_limit+length-1))
    #
    # results = []
    # for idx,g in enumerate(genres):
    #     print(g)
    #     res = model.evaluate([X_2[limits[idx][0]:limits[idx][1]],X_3[limits[idx][0]:limits[idx][1]]],y[limits[idx][0]:limits[idx][1]],batch_size=20)
    #     results.append(res)
