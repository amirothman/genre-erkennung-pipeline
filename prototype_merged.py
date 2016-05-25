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
import theano as T
from reshape_vector_one_vs_all import reshape_one_vs_all
from keras.utils.visualize_util import plot

batch_size = 5
nb_epoch = 10


print("creating model")
# create model
y = pickle.load(open("pickled_vectors/gztanmfcc_coefficients_label.pickle","rb"))
y_test = pickle.load(open("pickled_vectors/gztanmfcc_coefficients_evaluation_label.pickle","rb"))
# X_1 = pickle.load(open("pickled_vectors/tempotracker_tempo_training_vector.pickle","rb"))
# X_test_1 = pickle.load(open("pickled_vectors/tempotracker_tempo_evaluation_training_vector.pickle","rb"))

# model_1 = tempotrack_model.tempotrack_model((X_1.shape[1],X_1.shape[2]))

X_1 = pickle.load(open("pickled_vectors/gztanmfcc_coefficients_training_vector.pickle","rb"))
X_test_1 = pickle.load(open("pickled_vectors/gztanmfcc_coefficients_evaluation_training_vector.pickle","rb"))


X_2 = pickle.load(open("pickled_vectors/gztanspectral-contrast_peaks_training_vector.pickle","rb"))
X_test_2 = pickle.load(open("pickled_vectors/gztanspectral-contrast_peaks_evaluation_training_vector.pickle","rb"))

X_1_reshaped,y_reshaped = reshape_one_vs_all(X_1,y)
X_2_reshaped,_ = reshape_one_vs_all(X_2,y)

X_test_1_reshaped,y_test_reshaped = reshape_one_vs_all(X_test_1,y_test)
X_test_2_reshaped,_ = reshape_one_vs_all(X_test_2,y_test)

for x_1,x_2,x_1_test,x_2_test,y,y_test in zip(X_1_reshaped,X_2_reshaped,X_test_1_reshaped,X_test_2_reshaped,y_reshaped,y_test_reshaped):

    # print(x_1[0].shape)
    # print(x_2[0].shape)
    # print(y[0].shape)
    model_1 = mfcc_model.mfcc_model((x_1[0].shape[1],x_1[0].shape[2]))
    model_2 = spectral_contrast_peaks_model.model((x_2[0].shape[1],x_2[0].shape[2]))


    merged = Merge([model_1,model_2],mode="concat")

    final_model = Sequential()
    final_model.add(merged)
    final_model.add(Dense(1000))
    final_model.add(Dense(1, activation='softmax'))
    final_model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )

# plot(model_1,to_file="model_1.png")
# plot(model_2,to_file="model_2.png")
# plot(final_model,to_file="merged_model.png")



# json_string = final_model.to_json()
# with open("model_architecture/embeddings_10_sec_split_gztan_merged_model_architecture.json","w") as f:
#     f.write(json.dumps(json_string, sort_keys=True,indent=4, separators=(',', ': ')))

    # print("Fitting")
# #
# final_model.load_weights("model_weights/embeddings_10_sec_split_gztan_merged_model_weights.hdf5")
# #
# # for i in range(10):
# #     print("epoch",i)

# vector_length = int(len(X_2)/2)


    history = final_model.fit([x_1[0],x_2[0]], y[0],
                                batch_size=batch_size,
                                nb_epoch=nb_epoch,
                                validation_data=([x_1_test[0],x_2_test[0]], y_test[0]),
                                shuffle="batch"
                                )

# final_model.save_weights("model_weights/new_gztan_merged_model_weights.hdf5",overwrite=True)
#
# with open("experimental_results.json","w") as f:
#     f.write(json.dumps(history.history, sort_keys=True,indent=4, separators=(',', ': ')))


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
