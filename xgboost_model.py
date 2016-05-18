import numpy as np
np.random.seed(1337)
import pickle
import xgboost as xgb
from sklearn.cross_validation import KFold, train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score
from keras.layers import Dense, Dropout, Activation, Merge
import itertools
import json
from keras.models import Sequential, model_from_json
import mfcc_model
import spectral_contrast_peaks_model

y_test = pickle.load(open("pickled_vectors/10_sec_split_gztan_mfcc_coefficients_evaluation_label.pickle","rb"))

X_test_2 = pickle.load(open("pickled_vectors/10_sec_split_gztan_mfcc_coefficients_evaluation_training_vector.pickle","rb"))

X_test_3 = pickle.load(open("pickled_vectors/10_sec_split_gztan_spectral-contrast_peaks_evaluation_training_vector.pickle","rb"))

y = pickle.load(open("pickled_vectors/10_sec_split_gztan_mfcc_coefficients_label.pickle","rb"))
X_2 = pickle.load(open("pickled_vectors/10_sec_split_gztan_mfcc_coefficients_training_vector.pickle","rb"))
X_3 = pickle.load(open("pickled_vectors/10_sec_split_gztan_spectral-contrast_peaks_training_vector.pickle","rb"))
print("y",y.shape)
# print("X_1",X_1.shape)
# print("X_test_1",X_test_1.shape)
print("X_2",X_2.shape)
print("X_3",X_3.shape)


json_string = json.load(open("model_architecture/embeddings_10_sec_split_gztan_merged_model_architecture.json","r"))

model = model_from_json(json_string)

model.load_weights("model_weights/embeddings_10_sec_split_gztan_merged_model_weights.hdf5")

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )

old_weights = model.get_weights()

model_2 = mfcc_model.mfcc_model((X_2.shape[1],X_2.shape[2]))
model_3 = spectral_contrast_peaks_model.model((X_3.shape[1],X_3.shape[2]))
merged = Merge([model_2,model_3],mode="concat")


final_model = Sequential()
final_model.add(merged)
final_model.add(Dense(1000))
final_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )

new_weights = final_model.get_weights()
final_model.set_weights(old_weights[:len(new_weights)])

print("create embeddings")

vector_length = int(len(X_2)/2)
awesome_embeddings = final_model.predict([X_2[vector_length+1:],X_3[vector_length+1:]])

awesome_embeddings_test = final_model.predict([X_test_2,X_test_3])

y_categories = [list(el).index(1) for el in y][vector_length+1:]

y_test_categories = [list(el).index(1) for el in y_test]


print("fitting")
xgb_model = xgb.XGBClassifier().fit(awesome_embeddings,y_categories)
print("predicting")
predictions = xgb_model.predict(awesome_embeddings_test)

print(accuracy_score(y_test_categories, predictions))
