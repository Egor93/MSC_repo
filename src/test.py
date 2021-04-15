import sys
import os
from netCDF4 import Dataset
import matplotlib.pylab as plt
import numpy as np
from sklearn import tree
from matplotlib.colors import LogNorm
import time
start_time = time.time()
import os
from sklearn import ensemble
from IPython.core.display import display, HTML
import math


def func_plot_cfhist(dat_string, goal_var, input_var, train_fraction=0.6, rel_flag=0, max_depth_in=None):
    # TODO:rename train_fraction, its actually eval fraction
    # Applies a decisiontreeregressor returns a 2d hist plots
    # rel_flag: Since relative humidity is not part of the douze data it is calculated here and added according to this flat
    # rel_flag = 1, adds rel_hum
    # rel_flag = 2, adds qvlm_sqsm (qt-sat)
    # TODO: get rid of default rel_flag=0!! it will cause an error

    # Loading data
    data = Dataset(dat_string)
    # .ravel returns a masked flattened array
    # but all values are actually unmasked at this point

    # FLATTEN INPUT 2D ARRAY TO 1D MASKED ARRAY
    # TODO: add user-defined function flatten , flatten(variable)
    # y - e.g. LES cloud fraction,'True' values to compare with
    y = data.variables[goal_var][:].ravel()

    # FOR ALL INPUT VARIABLES' NAMES
    for input in input_var:
        # FLATTEN ARRAY 2D -> 1D
        x_temp = data.variables[input][:].ravel()
        # if input is the first in a list input fields
        # TODO: ask Philipp whats the point of this conditional?, use enumerate instead?
        if input == input_var[0]:
            X = x_temp
        else:
            #vstack concacenates row array into 2d array, by putting one (1,N) on top of the other
            # APPEND  a new 1xN row on every step
            X = np.vstack([X, x_temp])

    input_var_tmp = input_var.copy()
    # print(input_var_tmp)

    # Add relative humidity
    if rel_flag == 1:
        # qvlm - mean liquid water + vapor
        # qsm  - mean saturation mixing ratio?
        rel_sat = data.variables['qvlm'][:].ravel() / data.variables['qsm'][:].ravel()
        X = np.vstack([X, rel_sat])
        input_var_tmp.append('rel_sat')
    if rel_flag == 2:
        # TODO: qvlm, qsm are hardcoded here! Make them an explicit function input!
        # qvlm - liquid and gas water content, [kg/kg]?
        # qvl_qs the saturation deficit when qvlm is smaller than qsm
        qvl_qs = data.variables['qvlm'][:].ravel() - data.variables['qsm'][:].ravel()
        X = np.vstack([X, qvl_qs])
        input_var_tmp.append('qvl_qs')

    # print(input_var_tmp)
    # TODO: train_fraction is a fraction of true cloud fraction LES values to train on?
    n_e = int(len(y) * train_fraction)
    # or return a permuted range len=
    p = np.random.permutation(len(y))

    # TODO: variables below declared, but not called. Delete?
    y_shuf = y[p]
    X_shuf = X[:, p]

    #TODO: pick the train part randomly(not succesively) from y vector?

    y_train = y[n_e:]
    y_eval = y[:n_e]

    X_train = X[:, n_e:]
    X_eval = X[:, :n_e]

    X_train = X_train.transpose()
    X_eval = X_eval.transpose()

    # Now we set up and train the decision tree regressor
    # max_depth – The maximum depth of the tree. If None, then nodes are expanded
    # until all leaves are pure or until all leaves contain less than min_samples_split samples.
    regr_1 = tree.DecisionTreeRegressor(max_depth=max_depth_in)
    # Build a decision tree regressor from the training set (X, y)
    regr_1.fit(X_train, y_train)

    # Now the prediction magic happens
    y_pred = regr_1.predict(X_eval)
    # TODO: actually use array masking to increase performance. y_pred is a masked array, but has no masked entries!
    fig = plt.figure(figsize=(10, 10))
    bla = plt.hist2d(y_eval, y_pred, bins=100, norm=LogNorm())
    plt.colorbar()
    # bla = plt.hist2d(rel_sat,cl_l,bins=100, norm=LogNorm())
    # bla = plt.plot(sat_x,sat_ave,color='w',linewidth=2)
    # plt.vlines(1,0,1,'k')

    return fig



file_dir = '/home/igor/UNI/Master Project/Script/Data/'
dat_string = file_dir + 'ncr_pdf_douze_0125deg.nc'

goal_var = 'cl_l'
input_var = ['qsm', 'qtm', 'qlm', 'skew_l', 'var_l', 'var_t', 'tm', 'pm']
fig = func_plot_cfhist(dat_string, goal_var, input_var, rel_flag=2, max_depth_in=None)
plt.show()
print("--- %s seconds ---" % (time.time() - start_time))