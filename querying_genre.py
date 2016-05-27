import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import model_from_json, Sequential
import json
from extract_features import extract_features
from split_30_seconds_mono import batch_thirty_seconds,to_mono
from parse_songs import create_data_set
import os
import re
from numpy import genfromtxt
from keras.preprocessing import sequence

def process(song_name_without_ending,song_folder,file_format):
    to_mono(song_name_without_ending,file_format)
    batch_thirty_seconds(song_folder,file_format)

def extract(song_folder):
    extract_features(song_folder)

json_string = json.load(open("model_architecture/merged_model_architecture.json","r"))

model = model_from_json(json_string)
model.load_weights("model_weights/merged_model_weights.hdf5")
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )
# Parse song
song_name_without_ending = "dataset/query/song_1/song_1"
file_format = "mp3"
song_folder = "dataset/query/song_1"

process(song_name_without_ending,song_folder,file_format)
extract(song_folder)

keyword_2 = "mfcc_coefficients"

X_2 = []
for root, dirs, files in os.walk(song_folder, topdown=False):
    for name in files:
        if re.search("{0}.csv".format(keyword_2),name):
            song_path = (os.path.join(root,name))
            print(song_path)

            song_features = genfromtxt(song_path, delimiter=",")

            if len(song_features.shape) is 2:
                song_features = np.array([ _line[1:] for _line in song_features])
            elif len(song_features.shape) is 1:
                song_features = np.array([song_features[1:]])

            X_2.append(song_features)

mfcc_max_len = 0

with( open("maxlen_mfcc_coefficients","r") ) as _f:
    mfcc_max_len = int(_f.read())

X_2 = sequence.pad_sequences(X_2, maxlen=mfcc_max_len,dtype='float32')

keyword_3 = "spectral-contrast_peaks"
X_3 = []
for root, dirs, files in os.walk(song_folder, topdown=False):
    for name in files:
        if re.search("{0}.csv".format(keyword_3),name):
            song_path = (os.path.join(root,name))
            print(song_path)
            song_features = genfromtxt(song_path, delimiter=",")

            if len(song_features.shape) is 2:
                song_features = np.array([ _line[1:] for _line in song_features])
            elif len(song_features.shape) is 1:
                song_features = np.array([song_features[1:]])

            X_3.append(song_features)

spectral_max_len = 0
with( open("maxlen_spectral-contrast_peaks","r") ) as _f:
    spectral_max_len = int(_f.read())

X_3 = sequence.pad_sequences(X_3, maxlen=spectral_max_len,dtype='float32')

# print(len(X_2))
# print(len(X_3))

predictions = model.predict_classes([X_2,X_3])
print(predictions)
