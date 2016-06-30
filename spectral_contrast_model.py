from modelling import randomized_grid_search
import numpy as np
np.random.seed(1337)  # for reproducibility
import pickle

if __name__=="__main__":

    X_1 = pickle.load(open("pickled_vectors/3_generes_spectral-contrast_peaks_training_vector.pickle","rb"))

    X_2 = pickle.load(open("pickled_vectors/3_generes_spectral-contrast_valleys_training_vector.pickle","rb"))

    X = np.subtract(X_1,X_2)
    y = pickle.load(open("pickled_vectors/3_generes_spectral-contrast_peaks_label.pickle","rb"))

    X_1_test = pickle.load(open("pickled_vectors/3_generes_spectral-contrast_peaks_evaluation_training_vector.pickle","rb"))

    X_2_test = pickle.load(open("pickled_vectors/3_generes_spectral-contrast_valleys_evaluation_training_vector.pickle","rb"))

    X_test = np.subtract(X_1_test,X_2_test)

    y_test = pickle.load(open("pickled_vectors/3_generes_spectral-contrast_peaks_evaluation_label.pickle","rb"))


    # print(read_calculated_hyperparameters())

    nb_epoch = 3
    batch_size = 50

    # grid_search
    params = {}

    params["nb_filter"] = [300,200,100,80,60,50,40,20]
    params["filter_length"] = [1,2,5,10,20]
    params["pool_length"] = [1,2,3,4,5]
    params["subsample_length"] = [1,2,4,8,16]
    params["lstm_unit"] = [128,64,32,16,8]

    randomized_grid_search(X,X_test,
                           y,y_test,
                           nb_epoch,batch_size,
                           params = params,
                           results_file="spectral_contrast_results_1dcnn_lstm_3_generes.csv",
                           inner_loop=1)
