#!/usr/local/bin/python3

import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D,Convolution1D, MaxPooling1D
from keras.layers import LSTM, GRU, Flatten,BatchNormalization
import matplotlib.pyplot as plt
import pickle
import json
import pandas as pd
from random import shuffle,choice
from scipy.stats import spearmanr

def mfcc_model_cnn_2d(input_shape, nb_filter=100,
               nb_row=50, nb_col=2,
               pool_size=(50,5)):

    #CNN
    # nb_filter = 100
    # nb_row = 50
    # nb_col = 2

    # pool_size = (50,5)
    # subsample = (2,2)

    print("nb_filter ",nb_filter)
    print("nb_row ",nb_row)
    print("nb_col ",nb_col)
    print("pool_size ",pool_size)
    # print("subsample ",subsample)

    # create model
    model = Sequential()

    model.add(Convolution2D(
                            input_shape=input_shape,
                            nb_filter=nb_filter,
                            nb_row=nb_row,
                            nb_col = nb_col,
                            border_mode='valid',
                            # subsample=subsample
                            ))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.2))



    model.add(Flatten())

    # model.add(LSTM(512,init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one', activation='sigmoid', inner_activation='hard_sigmoid',dropout_W=0.2, dropout_U=0.2))

    return model

def mfcc_model_cnn_1d(input_shape, nb_filter=200,
                      filter_length=5,pool_length=10,
                      subsample_length=2,lstm_unit=32,gru_unit=32):

    #CNN
    # nb_filter = 100
    # nb_row = 50
    # nb_col = 2

    # pool_size = (50,5)
    # subsample = (2,2)
    #
    print("nb_filter ",nb_filter)
    print("filter_length ",filter_length)
    print("pool_length ",pool_length)
    print("subsample_length ",subsample_length)
    # print("lstm_unit ",lstm_unit)
    # print("gru_unit ",gru_unit)
    # create model

    # nb_filter_1 = 200
    # filter_length_1 = 5
    # pool_length_1 = 10
    # subsample_length_1 = 2

    model = Sequential()
    # model.add(BatchNormalization(input_shape=input_shape,axis=1))
    model.add(Convolution1D(
                            input_shape=input_shape,
                            nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            subsample_length=subsample_length))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(Dropout(0.2))
    model.add(Flatten())
    # #
    # model.add(LSTM(lstm_unit))
    # model.add(Convolution1D(
    #                         nb_filter=nb_filter,
    #                         filter_length=filter_length,
    #                         border_mode='valid',
    #                         subsample_length=subsample_length))
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(pool_length=pool_length))
    # model.add(Dropout(0.2))


    return model

def read_calculated_hyperparameters(file_path="results.txt"):
    lines = []
    with open(file_path,"r") as f:
        lines = f.read().split("\n")[1:]

    # model.add(Flatten())
    # model.add(LSTM(lstm_output_size))
    model.add(Dropout(0.4))
    model.add(Dense(numGenres))
    model.add(Dropout(0.2))
    #
    # model.add(Convolution1D(
    #                         nb_filter=int(nb_filter/10),
    #                         filter_length=int(filter_length/5),
    #                         border_mode='valid',
    #                         subsample_length=1))
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(pool_length=pool_length))
    # model.add(Dropout(0.4))

    lines =  (l.split(",") for l in lines)
    lines = (el[:-1] for el in lines)
    lines = ((int(el) for el in line) for line in lines)

    lines = [tuple(line) for line in lines]

    return lines

def randomized_grid_search_1d(X,X_test,y,y_test,nb_epoch,batch_size,results_file = "results_1d.csv",inner_loop=2):


    if not os.path.exists(results_file):
        with open(results_file,"w") as f:
            f.write("nb_filter,filter_length,pool_length,subsample_length,val_acc\n")

    params = {}

    params["nb_filter"] = [300,200,100,80,60,50,40,20]
    params["filter_length"] = [1,2,5,10,20]
    params["pool_length"] = [1,2,3,4,5]
    params["subsample_length"] = [1,2,4,8,16]
    # params["nb_filter"] = [300]
    # params["filter_length"] = [10]
    # params["pool_length"] = [80]
    # params["subsample_length"] = [2]
    # params["lstm_unit"] = [64,32,16,8]
    # params["gru_unit"] = [64,32,16,8]

    outer_loop = sum([len(p) for p in params.values()])
    for jdx in range(outer_loop):

        results_1d_temp_df = pd.read_csv(results_file)


        spearmanrs = []

        for column in results_1d_temp_df.columns[:-1]:
            idx_acc = results_1d_temp_df.sort_values("val_acc",ascending=False).index
            idx_val = results_1d_temp_df.sort_values(column,ascending=False).index
            spearmanrs.append((column,spearmanr(idx_acc,idx_val).correlation))

        spearmanrs.sort(key=lambda x: x[1],reverse=False)


        _n = 0
        while len(params[spearmanrs[_n][0]]) < 2:
            _n += 1

        param_list = []
        for p in params[spearmanrs[_n][0]]:
            df_temp = results_1d_temp_df[results_1d_temp_df[spearmanrs[_n][0]] == p]
            if not df_temp.empty:
                param_list.append((np.mean(df_temp.sort_values("val_acc").head(1).val_acc.values),p))
        print("param_list")
        print(param_list)
        if len(param_list) > 0 and jdx > 0:
            param_list.sort()
            worst_param = param_list[0][1]
            print("worst_param",spearmanrs[_n][0],worst_param)
            if int(worst_param) in params[spearmanrs[_n][0]]:
                params[spearmanrs[_n][0]].remove(int(worst_param))

        nb_filters = params["nb_filter"]
        filter_lengths = params["filter_length"]
        pool_lengths = params["pool_length"]
        subsample_lengths = params["subsample_length"]
        # lstm_units = params["lstm_unit"]
        # gru_units = params["gru_unit"]

        print(params)

        for idx in range(inner_loop):
            nb_filter = choice(nb_filters)
            filter_length = choice(filter_lengths)
            pool_length = choice(pool_lengths)
            subsample_length = choice(subsample_lengths)
            # lstm_unit = choice(lstm_units)
            # gru_unit = choice(gru_units)

            calculated_hyperparameters = read_calculated_hyperparameters(results_file)

            if (nb_filter,filter_length,pool_length,subsample_length) in calculated_hyperparameters:
                print("nb_filter:",nb_filter,"filter_length:",filter_length,"pool_length:",pool_length,"subsample_lengths:",subsample_length)
                print("parameter already calculated")
                continue
            try:
                model = mfcc_model_cnn_1d(tuple(X.shape[1:]), nb_filter=nb_filter,filter_length=filter_length,pool_length=pool_length,subsample_length=subsample_length)
                model.add(Dense(10))
                model.add(Activation('softmax'))
                model.compile(loss='categorical_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy']
                              )
                history = model.fit(X, y,
                          batch_size=batch_size,
                          nb_epoch=nb_epoch,
                          validation_data=(X_test, y_test),
                          shuffle=True
                        )
            except Exception as e:
                print(e)
                continue
            results = history.history

            val_acc = results["val_acc"][-1]

            with(open(results_file,"a")) as _f:
                _f.write("{0},{1},{2},{3},{4}\n".format(nb_filter,filter_length,pool_length,subsample_length,val_acc))

        if len(params.items()) is sum([len(v) for k,v in params.items()]):
            print("len",len(params.items()))

            print("sum",[len(v) for k,v in params.items()])
            print("finish em")
            break

def randomized_grid_search_2d(X,X_test,y,y_test,nb_epoch,batch_size,results_file = "results_2d.csv",outer_loop=25,inner_loop=2):



    if not os.path.exists(results_file):
        with open(results_file,"w") as f:
            f.write("nb_filter,nb_row,nb_col,pool_size_0,pool_size_1,val_acc\n")

    params = {}

    params["nb_filter"] = [150,100,50]
    params["nb_row"] = [20,40,60,80,100,120]
    params["nb_col"] = [2,4,6,8,10]
    params["pool_size_0"] = [25,50,75,100,125,150]
    params["pool_size_1"] = [1,5,10,15]

    for jdx in range(outer_loop):



        results_1d_temp_df = pd.read_csv(results_file)
        spearmanrs = []

        for column in results_1d_temp_df.columns[:-1]:
            idx_acc = results_1d_temp_df.sort_values("val_acc",ascending=False).index
            idx_val = results_1d_temp_df.sort_values(column,ascending=False).index
            spearmanrs.append((column,spearmanr(idx_acc,idx_val).correlation))

        spearmanrs.sort(key=lambda x: x[1],reverse=False)

        _n = 0
        while len(params[spearmanrs[_n][0]]) < 2:
            _n += 1

        param_list = []
        for p in params[spearmanrs[_n][0]]:
            df_temp = results_1d_temp_df[results_1d_temp_df[spearmanrs[_n][0]] == p]
            if not df_temp.empty:
                param_list.append((np.mean(df_temp.sort_values("val_acc").head(1).val_acc.values),p))
        print("param_list")
        print(param_list)
        if len(param_list) > 0:
            param_list.sort()
            worst_param = param_list[0][1]
            print("worst_param",spearmanrs[_n][0],worst_param)
            if int(worst_param) in params[spearmanrs[_n][0]]:
                params[spearmanrs[_n][0]].remove(int(worst_param))

        nb_filters = params["nb_filter"]
        nb_rows = params["nb_row"]
        nb_cols = params["nb_col"]
        pool_sizes_0 = params["pool_size_0"]
        pool_sizes_1 = params["pool_size_1"]
        print(params)

        for idx in range(inner_loop):
            nb_filter = choice(nb_filters)
            nb_row = choice(nb_rows)
            nb_col = choice(nb_cols)
            pool_size_0 = choice(pool_sizes_0)
            pool_size_1 = choice(pool_sizes_1)

            calculated_hyperparameters = read_calculated_hyperparameters(results_file)

            if (nb_filter,nb_row,nb_col,pool_size_0,pool_size_1) in calculated_hyperparameters:
                print(nb_filter,nb_row,nb_col,pool_size_0,pool_size_1)
                print("parameter already calculated")
                continue
            try:
                model = mfcc_model_cnn_2d((1,X.shape[2],X.shape[3]),nb_filter=nb_filter,
                                    nb_row=nb_row, nb_col=nb_col,
                                    pool_size=(pool_size_0,pool_size_1),
                                    )
                model.add(Dense(10))
                model.add(Activation('softmax'))
                model.compile(loss='categorical_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy']
                              )

                history = model.fit(X, y,
                          batch_size=batch_size,
                          nb_epoch=nb_epoch,
                          validation_data=(X_test, y_test),
                          shuffle=True
                        )
            except Exception as e:
                print(e)
                continue

            results = history.history

            val_acc = results["val_acc"][-1]

            with(open(results_file,"a")) as _f:
                _f.write("{0},{1},{2},{3},{4},{5}\n".format(nb_filter,nb_row,nb_col,pool_size_0,pool_size_1,val_acc))

        if len(params.items()) is sum([len(v) for k,v in params.items()]):
            print("len",len(params.items()))

            print("sum",[len(v) for k,v in params.items()])
            print("finish em")
            break

if __name__=="__main__":



    X = pickle.load(open("pickled_vectors/3_genre_mfcc_coefficients_training_vector.pickle","rb"))
    y = pickle.load(open("pickled_vectors/3_genre_mfcc_coefficients_labels.pickle","rb"))

    X_test = pickle.load(open("pickled_vectors/3_genre_mfcc_coefficients_evaluation_training_vector.pickle","rb"))
    y_test = pickle.load(open("pickled_vectors/3_genre_mfcc_coefficients_evaluation_labels.pickle","rb"))


    # print(read_calculated_hyperparameters())

    nb_epoch = 10
    batch_size = 50

    # grid_search

    # randomized_grid_search_1d(X,X_test,
    #                           y,y_test,
    #                           nb_epoch,batch_size,
    #                           results_file="results_1dcnn_1dcnn.csv",
    #                           inner_loop=100)

    # nb_filter = 200
    # filter_length = 10
    # pool_length = 1
    # subsample_length = 16

    # gru_unit = 64
    lstm_unit = 128
    # for lstm_unit in [512,256,128,64,32]:

    model = mfcc_model_cnn_1d(tuple(X.shape[1:]))
    model.add(Dense(3))

    batch_size = 20
    nb_epoch = 50

    model = mfcc_model((X.shape[1],X.shape[2]))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )
    history = model.fit(X, y,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, y_test),
              shuffle=True
            )

    for k,v in history.history.items():
        _keys = list(history.history.keys())
        _keys.sort()
        plt.subplot(411+_keys.index(k))
        plt.title(k)

        plt.plot(range(0,len(v)),v,marker="8",linewidth=1.5)
    if not os.path.exists("model_weights"):
        os.makedirs("model_weights")
    model.save_weights("model_weights/mfcc_model_weights.hdf5",overwrite=True)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()

    # 2d

    # X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
    #
    # X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
    #
    #
    # randomized_grid_search_2d(X,X_test,y,y_test,nb_epoch,batch_size,results_file = "results_2d_2.csv",outer_loop=25,inner_loop=2)
