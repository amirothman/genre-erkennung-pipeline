#!/usr/local/bin/python3
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Merge, Dense
import matplotlib.pyplot as plt
import pickle
import json
import os
from modelling import model_cnn_1d_lstm


batch_size = 50
nb_epoch = 20


print("load vectors")

y = pickle.load(open("pickled_vectors/3_genres_no_lonelymfcc_coefficients_label.pickle","rb"))
y_test = pickle.load(open("pickled_vectors/3_genres_no_lonelymfcc_coefficients_evaluation_label.pickle","rb"))

X_1 = pickle.load(open("pickled_vectors/3_genres_no_lonelymfcc_coefficients_training_vector.pickle","rb"))
X_test_1 = pickle.load(open("pickled_vectors/3_genres_no_lonelymfcc_coefficients_evaluation_training_vector.pickle","rb"))

X_2 = pickle.load(open("pickled_vectors/3_genres_no_lonelyspectral-contrast_peaks_training_vector.pickle","rb"))
X_test_2 = pickle.load(open("pickled_vectors/3_genres_no_lonelyspectral-contrast_peaks_evaluation_training_vector.pickle","rb"))

X_3 = pickle.load(open("pickled_vectors/3_genres_no_lonelyspectral-contrast_valleys_training_vector.pickle","rb"))
X_test_3 = pickle.load(open("pickled_vectors/3_genres_no_lonelyspectral-contrast_valleys_evaluation_training_vector.pickle","rb"))

# X_4 = pickle.load(open("pickled_vectors/3_genres_no_lonelyadaptivespectrogram_output_training_vector.pickle","rb"))
#
# X_4 = X_4[2:]
# X_test_4 = pickle.load(open("pickled_vectors/3_genres_no_lonelyadaptivespectrogram_output_evaluation_training_vector.pickle","rb"))



print("X_1.shape",X_1.shape)
print("X_2.shape",X_2.shape)
print("X_3.shape",X_3.shape)
# print("X_4.shape",X_4.shape)

print("X_test_1.shape",X_test_1.shape)
print("X_test_2.shape",X_test_2.shape)
print("X_test_3.shape",X_test_3.shape)
# print("X_test_4.shape",X_test_4.shape)

print("create model")

# Best parameters

# nb_filter,filter_length,pool_length,subsample_length,lstm_units

# mfcc (model 1)
# 60,10,2,16,128,0.78

# Spectral Contrast peaks (model 2)
# 50,10,3,8,0.6729589363576782

# Spectral Contrast valleys (model 3)
# 50,20,3,16,16,0.6352052823616233


model_1 = model_cnn_1d_lstm(tuple(X_1.shape[1:]),
                            nb_filter=60,
                            filter_length=10,
                            pool_length=2,
                            subsample_length=16,
                            lstm_units = 128)

model_2 = model_cnn_1d_lstm(tuple(X_2.shape[1:]),
                            nb_filter=50,
                            filter_length=10,
                            pool_length=3,
                            subsample_length=8,
                            lstm_units = 64)

model_3 = model_cnn_1d_lstm(tuple(X_3.shape[1:]),
                            nb_filter=50,
                            filter_length=20,
                            pool_length=3,
                            subsample_length=16,
                            lstm_units = 16)

# merged = Merge([model_1,model_2,model_3,model_4],mode="concat")
merged = Merge([model_1,model_2,model_3],mode="concat")
#
final_model = Sequential()
final_model.add(merged)
final_model.add(Dense(3, activation='softmax'))
final_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )

json_string = final_model.to_json()
with open("model_architecture/merged_model_architecture.json","w") as f:
    f.write(json.dumps(json_string, sort_keys=True,indent=4, separators=(',', ': ')))
#
# print("Fitting")
#
#
# history = final_model.fit([X_1,X_2,X_3], y,
#                             batch_size=batch_size,
#                             nb_epoch=nb_epoch,
#                             validation_data=([X_test_1,X_test_2,X_test_3], y_test),
#                             shuffle="batch"
#                             )
#
# if not os.path.exists("model_weights"):
#     os.makedirs("model_weights")
# final_model.save_weights("model_weights/super_awesome_merged_model_weights.hdf5",overwrite=True)
# #
# # with open("experimental_results.json","w") as f:
# #     f.write(json.dumps(history.history, sort_keys=True,indent=4, separators=(',', ': ')))
# #
# #
# for k,v in history.history.items():
#     _keys = list(history.history.keys())
#     _keys.sort()
#     plt.subplot(411+_keys.index(k))
#     plt.title(k)
#
#     plt.plot(range(0,len(v)),v,marker="8",linewidth=1.5)
#
# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
# plt.show()
