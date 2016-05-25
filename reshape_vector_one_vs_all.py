import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Merge
import matplotlib.pyplot as plt
import pickle
import json
import mfcc_model
import tempotrack_model
import spectral_contrast_peaks_model

def reshape_one_vs_all(x,y):
    reshaped_idx = []
    for i in range(len(y[0])):
        temp_y = [_idx for _idx,_y in enumerate(y) if i is list(_y).index(1.0) ]
        reshaped_idx.append(temp_y)

    reshaped_vector = []
    for indices in reshaped_idx:
        temp_x_vectors_positive = [ x[idx] for idx in indices]
        negative_indices = list(range(len(x)))

        for idx in indices:
            negative_indices.remove(idx)
        temp_x_vectors_negative = [ x[idx] for idx in negative_indices]
        reshaped_vector.append((temp_x_vectors_positive,temp_x_vectors_negative))

    x = [np.array([el[0]+el[1]]) for el in reshaped_vector]
    y = [np.array([[1]*len(el[0])+[0]*len(el[1])]) for el in reshaped_vector]

    return x,y

if __name__=="__main__":

    batch_size = 50
    nb_epoch = 100

    y = pickle.load(open("pickled_vectors/gztanmfcc_coefficients_label.pickle","rb"))
    y_test = pickle.load(open("pickled_vectors/gztanmfcc_coefficients_evaluation_label.pickle","rb"))
    X_2 = pickle.load(open("pickled_vectors/gztanmfcc_coefficients_training_vector.pickle","rb"))
    X_test_2 = pickle.load(open("pickled_vectors/gztanmfcc_coefficients_evaluation_training_vector.pickle","rb"))

    reshaped_x, reshaped_y = reshape_one_vs_all(X_2,y)
