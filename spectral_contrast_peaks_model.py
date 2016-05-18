import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import LSTM, GRU, Flatten
import matplotlib.pyplot as plt
import pickle
import json

# load vectorized song features
#
def model(input_shape):

    nb_filter = 100
    filter_length = 3
    hidden_dims = 250
    pool_length = 1

    # LSTM
    lstm_output_size = 100

    # print("creating model")
    # create model
    model = Sequential()
    #
    # model.add(Convolution1D(
    #                         input_shape=input_shape,
    #                         nb_filter=nb_filter,
    #                         filter_length=filter_length,
    #                         border_mode='valid',
    #                         subsample_length=4))
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(pool_length=pool_length))
    # model.add(Dropout(0.2))

    model.add(LSTM(lstm_output_size,
                    input_shape=input_shape,
                    activation='sigmoid',
                    inner_activation='hard_sigmoid',
                    # return_sequences=True
                    ))

    model.add(Dropout(0.2))


    #
    #
    # # #
    # # # #
    # model.add(Convolution1D(
    #                         nb_filter=int(nb_filter/5),
    #                         filter_length=int(filter_length/10),
    #                         border_mode='valid',
    #                         subsample_length=1))
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(pool_length=pool_length))
    # model.add(Dropout(0.2))

    # model.add(Flatten())
    # # #
    # # #
    #
    # model.add(Convolution1D(
    #                         nb_filter=int(nb_filter/10),
    #                         filter_length=int(filter_length/20),
    #                         border_mode='valid',
    #                         subsample_length=2))
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(pool_length=pool_length))
    # model.add(Dropout(0.2))
    #
    # model.add(LSTM(lstm_output_size,
    #                 # input_shape=(X.shape[1],X.shape[2]),
    #                 activation='sigmoid',
    #                 inner_activation='hard_sigmoid',
    #                 return_sequences=True
    #                 ))
    #
    #
    # #
    # # model.add(Dropout(0.2))
    #
    # model.add(Dropout(0.2))
    # model.add(Flatten())
    # model.add(Dense(10))
    # model.add(Dropout(0.2))
    # model.add(Flatten())
    # model.add(LSTM(lstm_output_size))
    return model

# model.add(LSTM(lstm_output_size,
#                 # input_shape=(X.shape[1],X.shape[2]),
#                 activation='sigmoid',
#                 inner_activation='hard_sigmoid',
#                 # return_sequences=True
#                 ))
#
# model.add(Dropout(0.2))
#
# model.add(Convolution1D(
#                         nb_filter=int(nb_filter/10),
#                         filter_length=int(filter_length/5),
#                         border_mode='valid',
#                         subsample_length=1))
# model.add(Activation('relu'))
# model.add(MaxPooling1D(pool_length=pool_length))
# model.add(Dropout(0.4))


# model.add(Lambda(max_1d, output_shape=(nb_filter)))
# model.add(LSTM(lstm_output_size))
# model.add(Dropout(0.2))
# # We add a vanilla hidden layer:
# model.add(Activation('relu'))
# model.add(Dense(hidden_dims))
# model.add(Flatten())
# model.add(Dense(200))
# model.add(Activation("sigmoid"))
# model.add(Dropout(0.2))

if __name__=="__main__":
    X = pickle.load(open("pickled_vectors/spectral-contrast_peaks_training_vector.pickle","rb"))
    y = pickle.load(open("pickled_vectors/spectral-contrast_peaks_label.pickle","rb"))

    X_test = pickle.load(open("pickled_vectors/spectral-contrast_peaks_evaluation_training_vector.pickle","rb"))
    y_test = pickle.load(open("pickled_vectors/spectral-contrast_peaks_evaluation_label.pickle","rb"))

    batch_size = 20
    nb_epoch = 20
    model = model((X.shape[1],X.shape[2]))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )



    # print(X)
    print("X shape",X.shape)
    print("y shape",y.shape)
    print("X_test", X_test.shape)
    print("y_test", y_test.shape)

    print("Fitting")
    #
    # X = np.random.random(X.shape)
    # y = np.random.random(y.shape)
    #
    history = model.fit(X, y,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, y_test),
              shuffle=True
            )
    #
    # # with open("experimental_results.json","w") as f:
    # #     f.write(json.dumps(history.history, sort_keys=True,indent=4, separators=(',', ': ')))
    #
    for k,v in history.history.items():
        # print(k,v)
        _keys = list(history.history.keys())
        _keys.sort()
        plt.subplot(411+_keys.index(k))
        # x_space = np.linspace(0,1,len(v))
        plt.title(k)

        plt.plot(range(0,len(v)),v,marker="8",linewidth=1.5)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()
