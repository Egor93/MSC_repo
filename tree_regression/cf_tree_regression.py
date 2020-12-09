from netCDF4 import Dataset
import matplotlib.pylab as plt
from netCDF4 import Dataset
import numpy as np
from sklearn import tree
from matplotlib.colors import LogNorm
import xarray as xr
import time

###################
# DATA PREPROCESSING
###################
class DataPrepro():

    def __init__(self, curdir, fname, goalvar,input_vars,add_vars,eval_fraction):
        self.curdir     = curdir
        self.fname      = fname
        self.goalvar    = goalvar
        self.input_vars = input_vars
        self.add_vars   = add_vars
        self.eval_fraction=eval_fraction


    def read_netcdf(self):

        filepath = self.curdir + self.fname
        # ds contains 63 fields
        self.ds = xr.open_dataset(filepath)

        return None

    def read_netcdf4(self):

        filepath = self.curdir + self.fname
        # ds contains 63 fields
        self.ds = Dataset(filepath).variables

        return None

    def proc_goalvar(self):
        goalvar_arr = self.ds[self.goalvar]
        self.goalvar_flat = goalvar_arr.data.flatten()

        return None

    def proc_goalvar4(self):
        goalvar_arr = self.ds[self.goalvar][:]
        #ravel -> from 3D(nx*ny, 1, 150) to 1D(nx*ny*150)
        self.goalvar_flat = goalvar_arr.ravel()

        return None

    def proc_inputvars(self):
        # returns X_arr - 2D arr 
        # of shape (n input_vars,nx*ny*150)
        rows=len(self.input_vars)
        columns= self.goalvar_flat.shape[0]
        X_arr=np.zeros((rows,columns))
        for index,var in enumerate(self.input_vars):
            var_arr = self.ds[var]
            X_arr[index]=var_arr.data.flatten()

        self.X_arr=X_arr

        return None

    def proc_inputvars4(self):
        # returns X_arr - 2D arr 
        # of shape (n input_vars,nx*ny*150)
        rows = len(self.input_vars)
        columns = self.goalvar_flat.shape[0]
        X_arr = np.zeros((rows, columns))
        for index, var in enumerate(self.input_vars):
            var_arr = self.ds[var][:]
            X_arr[index] = var_arr.ravel()

        self.X_arr = X_arr

        return None


    def proc_addvars(self):
        assert len(self.add_vars)==2
        qvlm_arr=self.ds[self.add_vars[0]].data
        qvlm_arr_flat=qvlm_arr.flatten()

        qsm_arr = self.ds[self.add_vars[1]].data
        qsm_arr_flat = qsm_arr.flatten()
        qvl_qs = qvlm_arr_flat - qsm_arr_flat
        self.X_arr=np.vstack([self.X_arr,qvl_qs])

    def proc_addvars4(self):
        assert len(self.add_vars) == 2
        qvlm_arr = self.ds[self.add_vars[0]][:]
        qvlm_arr_flat = qvlm_arr.ravel()

        qsm_arr = self.ds[self.add_vars[1]][:]
        qsm_arr_flat = qsm_arr.ravel()
        qvl_qs = qvlm_arr_flat - qsm_arr_flat
        self.X_arr = np.vstack([self.X_arr, qvl_qs])

        return None

    def split_data(self):
        assert self.eval_fraction<1
        total_len=self.goalvar_flat.shape[0]
        fractional_len = int(total_len * self.eval_fraction)
        # split goal variable
        self.goalvar_train = self.goalvar_flat[fractional_len:]
        self.goalvar_eval = self.goalvar_flat[:fractional_len]
        # split input variable
        self.X_train = self.X_arr[:, fractional_len:]
        self.X_eval  = self.X_arr[:, :fractional_len]

        self.X_train = self.X_train.transpose()
        self.X_eval = self.X_eval.transpose()

    def get_processed_data(self):
        data_dict={}
        # PACK INPUT VARS
        data_dict['X_train']=self.X_train
        data_dict['X_eval'] = self.X_eval
        # PACK VAR
        data_dict['goalvar_train'] = self.goalvar_train
        data_dict['goalvar_eval'] = self.goalvar_eval

        return data_dict


def regression(regtype,processed_data,max_depth_in):
    # UNPACK INPUT VARS
    X_train=processed_data['X_train']
    X_eval = processed_data['X_eval']
    # UNPACK GOAL VARS
    goalvar_train = processed_data['goalvar_train']
    goalvar_eval = processed_data['goalvar_eval']
    if regtype == 'decision_tree':
        # TRAINING
        regtree=tree.DecisionTreeRegressor(max_depth=max_depth_in)
        regtree.fit(X_train, goalvar_train)
        # PREDICTION
        goal_var_pred = regtree.predict(X_eval)

    return goal_var_pred,goalvar_eval


def hist_plot(goal_var_pred,goalvar_eval):
    fig = plt.figure(figsize=(10, 10))
    plt.hist2d(goalvar_eval, goal_var_pred, bins=100, norm=LogNorm())
    plt.colorbar()
    plt.show()

    return None


def main():
    # DATA PREPROCESSING
    prepro=DataPrepro(curdir,fname,goal_var,input_vars,add_vars,eval_fraction)
    # methods should be in this particular order!!
    # prepro.read_netcdf()
    prepro.read_netcdf4()
    # prepro.proc_goalvar()
    prepro.proc_goalvar4()
    # prepro.proc_inputvars()
    prepro.proc_inputvars4()
    # prepro.proc_addvars()
    prepro.proc_addvars4()
    prepro.split_data()
    processed_data = prepro.get_processed_data()

    # REGRESSION
    goal_var_pred, goalvar_eval = regression(regtype,processed_data,max_depth_in)

    # VISUALISATION
    hist_plot(goal_var_pred, goalvar_eval)
    print('wow')



if __name__=='__main__':
    # DEFAULT PARAMETERS - global variables
    curdir='/home/igor/UNI/Master Project/Script/Data/'
    fname='ncr_pdf_douze_0125deg.nc'
    goal_var = 'cl_l'
    input_vars = ['qsm', 'qtm', 'qlm', 'skew_l', 'var_l', 'var_t', 'tm', 'pm']
    add_vars = ['qvlm','qsm']
    eval_fraction=0.6
    regtype = 'decision_tree'
    max_depth_in=None

    start_time = time.time()
    main()
    end_time =time.time()
    print("--- %s seconds ---" % (start_time - end_time))
