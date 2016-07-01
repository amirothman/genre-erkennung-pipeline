import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import model_from_json, Sequential
import json
from parse_features import get_vector
from extract_features import extract_features_single,extract_features
from keras.preprocessing import sequence
from split_30_seconds import ten_seconds

_file = "query/song_8/soulja_boy.mp3"
_folder_path = "query/song_8/"

# Split

# ten_seconds(_file)

# Extract Features

_features = [
              "vamp:qm-vamp-plugins:qm-mfcc:coefficients",
              "vamp:bbc-vamp-plugins:bbc-spectral-contrast:peaks",
              "vamp:bbc-vamp-plugins:bbc-spectral-contrast:valleys"
              ]

# extract_features(_folder_path,_features)

# Create Vectors

_keywords = ["qm-mfcc_coefficients","bbc-spectral-contrast_peaks","bbc-spectral-contrast_valleys"]

X_1_unpadded = get_vector(_folder_path,_keywords[0],lower_limit=1)
X_2_unpadded = get_vector(_folder_path,_keywords[1],lower_limit=1)
X_3_unpadded = get_vector(_folder_path,_keywords[2],lower_limit=1)

# Pad vectors

X_1 = sequence.pad_sequences(X_1_unpadded, maxlen=470,dtype='float32')
X_2 = sequence.pad_sequences(X_2_unpadded, maxlen=940,dtype='float32')
X_3 = sequence.pad_sequences(X_3_unpadded, maxlen=940,dtype='float32')

# Build Model

json_string = json.load(open("model_architecture/merged_model_architecture.json","r"))
model = model_from_json(json_string)
model.load_weights("model_weights/super_awesome_merged_model_weights.hdf5")
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )

genres = ["Hip Hop", "Pop", "Rock"]
genres_categorical_vector = [[1,0,0],[0,1,0],[0,0,1]]
predictions = model.predict([X_1,X_2,X_3])
print(predictions)
# results = [genres_categorical_vector[np.argmax(p)] for p in predictions]
print(results)
counts = np.bincount(results)
print(genres[np.argmax(counts)])
