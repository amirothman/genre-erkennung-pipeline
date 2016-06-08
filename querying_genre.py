#!/usr/local/bin/python3
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import model_from_json, Sequential
import json
from extract_features import extract_features
from split_30_seconds import batch_thirty_seconds,to_mono
import os
import re
import sys
from numpy import genfromtxt
from keras.preprocessing import sequence

if not os.path.exists("model_weights/merged_model_weights.hdf5"):
    print("No model weights found in path 'model_weights/merged_model_weights.hdf5'")
else:
    json_string = json.load(open("model_architecture/merged_model_architecture.json","r"))
    model = model_from_json(json_string)
    model.load_weights("model_weights/merged_model_weights.hdf5")
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )
    # Parse song
    if len(sys.argv) < 2:
        print("missing parameter")
    else:
        
        filename = sys.argv[1]
        song_folder = os.path.dirname(os.path.realpath(filename))#should get the directory to the file

        #to_mono(filename)
        batch_thirty_seconds(song_folder)
        extract_features(song_folder)

        keyword_2 = "mfcc_coefficients"

        x2 = []
        for root, dirs, files in os.walk(song_folder, topdown=False):
            for name in files:
                if re.search(keyword_2+".csv",name):
                    song_path = (os.path.join(root,name))
                    print(song_path)

                    song_features = genfromtxt(song_path, delimiter=",")

                    if len(song_features.shape) is 2:
                        song_features = np.array([ _line[1:] for _line in song_features])
                    elif len(song_features.shape) is 1:
                        song_features = np.array([song_features[1:]])

                    x2.append(song_features)

        mfcc_max_len = 0

        with( open("maxlen_mfcc_coefficients","r") ) as _f:
            mfcc_max_len = int(_f.read())

        x2 = sequence.pad_sequences(x2, maxlen=mfcc_max_len,dtype='float32')

        keyword_3 = "spectral-contrast_peaks"
        x3 = []
        for root, dirs, files in os.walk(song_folder, topdown=False):
            for name in files:
                if re.search(keyword_3+".csv",name):
                    song_path = (os.path.join(root,name))
                    print(song_path)
                    song_features = genfromtxt(song_path, delimiter=",")

                    if len(song_features.shape) is 2:
                        song_features = np.array([ _line[1:] for _line in song_features])
                    elif len(song_features.shape) is 1:
                        song_features = np.array([song_features[1:]])

                    x3.append(song_features)

        spectral_max_len = 0
        with( open("maxlen_spectral-contrast_peaks","r") ) as _f:
            spectral_max_len = int(_f.read())

        x3 = sequence.pad_sequences(x3, maxlen=spectral_max_len,dtype='float32')

        predictions = model.predict_classes([x2,x3])
        print(predictions)
