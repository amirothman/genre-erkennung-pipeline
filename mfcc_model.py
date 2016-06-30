from modelling import randomized_grid_search
import numpy as np
np.random.seed(1337)  # for reproducibility
import pickle


if __name__=="__main__":

    X = pickle.load(open("pickled_vectors/3_generes_mfcc_coefficients_training_vector.pickle","rb"))
    y = pickle.load(open("pickled_vectors/3_generes_mfcc_coefficients_label.pickle","rb"))

    X_test = pickle.load(open("pickled_vectors/3_generes_mfcc_coefficients_evaluation_training_vector.pickle","rb"))
    y_test = pickle.load(open("pickled_vectors/3_generes_mfcc_coefficients_evaluation_label.pickle","rb"))


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
                              nb_epoch,
                              batch_size,
                              params = params,
                              results_file="mfcc_results_1dcnn_lstm_3_generes.csv",
                              inner_loop=1)
