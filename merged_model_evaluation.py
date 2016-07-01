#!/usr/local/bin/python3
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential, model_from_json
from keras.layers import Merge, Dense
from sklearn.metrics import precision_recall_curve,average_precision_score
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

json_string = json.load(open("model_architecture/merged_model_architecture.json","r"))

model = model_from_json(json_string)

model.load_weights("model_weights/super_awesome_merged_model_weights.hdf5")
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )

n_classes = 3
y_score = model.predict([X_test_1,X_test_2,X_test_3])
# Compute Precision-Recall and plot curve
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

# Compute micro-average ROC curve and ROC area
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
average_precision["micro"] = average_precision_score(y_test, y_score,average="micro")

# Plot Precision-Recall curve
plt.clf()
plt.plot(recall[0], precision[0], label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
plt.legend(loc="lower left")
plt.show()

genres = ["Hip Hop","Pop","Rock"]
# Plot Precision-Recall curve for each class
plt.clf()
plt.plot(recall["micro"], precision["micro"],
         label='micro-average Precision-recall curve (area = {0:0.2f})'
               ''.format(average_precision["micro"]))
for i in range(n_classes):
    plt.plot(recall[i], precision[i],
             label='Precision-recall curve of class {0} (area = {1:0.2f})'
                   ''.format(genres[i], average_precision[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")
plt.show()
