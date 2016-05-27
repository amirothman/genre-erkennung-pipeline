import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

"""
One box represents one song. This plot gives us hints whether the feature is good enough to differentiate the genres. This script samples different 10 songs (num_plots) from each genre and plot their respective audio feature.
"""

genres = ["blues","classical","country",
          "disco","hiphop","jazz",
          "metal","pop","reggae",
          "rock"]

num_plots = 10
# plt.close('all')
f, axarr = plt.subplots(10,num_plots, sharex=True,sharey=True)
rand_list = []
for jdx in range(num_plots):
    csv_list = []

    rand_num_1 = np.random.randint(0,2)
    rand_num_2 = np.random.randint(0,10)

    while (rand_num_1,rand_num_2) in rand_list:
        print("stuck?")
        rand_num_1 = np.random.randint(0,2)
        rand_num_2 = np.random.randint(0,10)

    rand_list.append((rand_num_1,rand_num_2))

    for genre in genres:
        csv_path = "../dataset/gztan/test/{0}/{0}.000{1}{2}_vamp_qm-vamp-plugins_qm-mfcc_coefficients.csv".format(genre,rand_num_1,rand_num_2)
        # csv_path = "../dataset/gztan/test/{0}/{0}.000{1}{2}_vamp_bbc-vamp-plugins_bbc-spectral-contrast_peaks.csv".format(genre,rand_num_1,rand_num_2)
        # csv_path = "../dataset/gztan/test/{0}/{0}.000{1}{2}_vamp_bbc-vamp-plugins_bbc-spectral-flux_spectral-flux.csv".format(genre,rand_num_1,rand_num_2)

        # mtg-melodia_melodia_melody.

        csv_list.append(genfromtxt(csv_path,delimiter=","))

    for idx,ax in enumerate(axarr):
        time_array = csv_list[idx][:,0]
        spectral = csv_list[idx][:,1:]
        axarr[idx][jdx].plot(time_array,spectral,"r.",alpha=0.45,ms=2.0)
        axarr[idx][jdx].set_axis_bgcolor('yellow')
        axarr[idx][jdx].axes.get_xaxis().set_visible(False)
        axarr[idx][jdx].set_yticklabels([])
        # if idx is 0:
        #     axarr[idx][jdx].set_title("{0}{1}".format(rand_num_1,rand_num_2))
        if jdx is 0:
            # print(idx)

            axarr[idx][jdx].set_ylabel(genres[idx])


plt.show()
