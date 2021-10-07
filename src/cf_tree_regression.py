import matplotlib.pylab as plt
from netCDF4 import Dataset
import numpy as np
from sklearn import tree,ensemble
from matplotlib.colors import LogNorm
import datetime
import logging
import configparser

def initLog():
    # create logger
    logger = logging.getLogger(__name__)
    # set ERROR to suppress debug/info messages, otherwise DEBUG
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('logger_file.txt')
    fh.setLevel(logging.DEBUG)
    # create console handler which logs even debug messages
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    #ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


###################
# DATA PREPROCESSING
###################
class DataPrepro():
    # DESCRIPTION
    #           Class for prepocessing of netcdf data.
    #           Preprocessed data then used in various regression methods.
    # INPUT
    #           eval_fraction -  fraction of input/goal variables for evaluation of regression method
    #                            train fraction = 1 - eval_fraction
    #           goalvar       -  variable which we try to approximate, LES 'True' values to compare with
    #           input_vars    -  LES 'observed' variables, used as regression algorithm input
    #                            to get the model variable - goalvar

    def __init__(self, abspath, vars_dict, eval_fraction, regtype, ML_max_depth, resolution):
        self.abspath         = abspath
        self.goalvar         = vars_dict['goal_var']
        self.input_vars      = vars_dict['input_vars']
        self.add_vars        = vars_dict['add_var']
        self.eval_fraction   = eval_fraction
        self.regtype         = regtype
        self.max_depth       = ML_max_depth
        self.resolution      = resolution



    def read_netcdf(self):

        filepath = self.abspath
        # ds contains 63 fields
        self.ds = Dataset(filepath).variables

        return None


    def missing_values_present(self,var):
        check_result=False
        # boolean array of missing variables
        netcdf_variable = self.ds[var]
        missing_value = netcdf_variable.missing_value
        # boolean array of missing variables
        missing_values_bool = netcdf_variable == missing_value
        
        # if there is any missing value present
        if True in missing_values_bool:
            check_result=True

        return check_result,missing_values_bool


    def proc_goalvar(self):
        goalvar_arr = self.ds[self.goalvar][:]
        #ravel -> from 3D(nx*ny, 1, 150) to 1D(nx*ny*150)
        self.goalvar_flat = goalvar_arr.ravel()

        return None


    def proc_inputvars(self):
        # returns X_arr - 2D arr 
        # of shape (n input_vars,nx*ny*150)
        rows = len(self.input_vars)
        columns = self.goalvar_flat.shape[0]
        # Create !masked! array to be filled.
        # otherwise mask will be lost while filling X_arr
        X_arr = np.ma.zeros((rows, columns))

        for index, var in enumerate(self.input_vars):
            # unpach method output into boolean and boolean array of missing variables
            positive_check_result,missing_values_bool=self.missing_values_present(var)
            if positive_check_result:
                # explicit masking of missing values
                # + print a user warning
                logger.warning("Missing values in {} field were masked!".format(var))
                var_unmasked = np.array(self.ds[var])
                var_arr=np.ma.masked_array(var_unmasked,missing_values_bool)
            else:
                # [:] converts netcdf4 to masked array
                var_arr = self.ds[var][:]

            X_arr[index] = var_arr.ravel()

        self.X_arr = X_arr

        return None

    def proc_addvars(self):
        assert len(self.add_vars) == 2
        qvlm_arr = self.ds[self.add_vars[0]][:]
        qvlm_arr_flat = qvlm_arr.ravel()

        qsm_arr = self.ds[self.add_vars[1]][:]
        qsm_arr_flat = qsm_arr.ravel()
        qvl_qs = qvlm_arr_flat - qsm_arr_flat
        self.X_arr = np.vstack([self.X_arr, qvl_qs])

        return None

    def split_data_sequentially(self):
        logger.info("splitting the input/output data sets sequentially")
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
        
    def split_data_randomly(self):
        logger.info("splitting the input/output data sets randomly")
    	 # SHUFFLE both input and output arrays before splitting
        assert self.eval_fraction<1
        total_len=self.goalvar_flat.shape[0]
        fractional_len = int(total_len * self.eval_fraction)
        train_len=total_len-fractional_len
        
        # shuffle output(goal) variable
        bool_arr=np.array([True]*train_len+[False]*fractional_len)
        np.random.shuffle(bool_arr)
        # split goal variable
        self.goalvar_train = self.goalvar_flat[bool_arr]
        self.goalvar_eval = self.goalvar_flat[:fractional_len]
        #invert boolean array values to get the rest of shuffled set
        bool_arr_invert=np.invert(bool_arr)
        self.goalvar_eval = self.goalvar_flat[bool_arr_invert]
        
        # shuffle input variable INDEPNDENTLY FROM INPUT VARIABLE
        #bool_arr=np.array([True]*train_len+[False]*fractional_len)
        #np.random.shuffle(bool_arr) 
        # split input variable
        self.X_train = self.X_arr[:,bool_arr]
        #invert boolean array values to get the rest of shuffled set
        #bool_arr_invert=np.invert(bool_arr)
        self.X_eval  = self.X_arr[:,bool_arr_invert]
        
        self.X_train = self.X_train.transpose()
        self.X_eval = self.X_eval.transpose()    
        
    def split_data_randomly_failed(self):
        logger.info("splitting the input/output data sets randomly")
    	 # SHUFFLE both input and output arrays before splitting
        assert self.eval_fraction<1
        total_len=self.goalvar_flat.shape[0]
        fractional_len = int(total_len * self.eval_fraction)
        train_len=total_len-fractional_len
        
        # shuffle output(goal) variable
        bool_arr=np.array([True]*train_len+[False]*fractional_len)
        np.random.shuffle(bool_arr)
        # split goal variable
        self.goalvar_train = self.goalvar_flat[bool_arr]
        self.goalvar_eval = self.goalvar_flat[:fractional_len]
        #invert boolean array values to get the rest of shuffled set
        bool_arr_invert=np.invert(bool_arr)
        self.goalvar_eval = self.goalvar_flat[bool_arr_invert]
        
        # shuffle input variable INDEPNDENTLY FROM INPUT VARIABLE
        bool_arr=np.array([True]*train_len+[False]*fractional_len)
        np.random.shuffle(bool_arr) 
        # split input variable
        self.X_train = self.X_arr[:,bool_arr]
        #invert boolean array values to get the rest of shuffled set
        bool_arr_invert=np.invert(bool_arr)
        self.X_eval  = self.X_arr[:,bool_arr_invert]
        
        self.X_train = self.X_train.transpose()
        self.X_eval = self.X_eval.transpose()
 

    def get_processed_data(self,split_randomly=True):

        logger.info(f'RESOLUTION={self.resolution}_degr   INPUTVAR={self.input_vars}   ADDVAR={self.add_vars}') 
        logger.info(f'REGTYPE={self.regtype}      ML_max_depth={self.max_depth}       EVAL_FRACTION={self.eval_fraction}')
        
        self.read_netcdf()
        self.proc_goalvar()
       #logger.info("Processed goal vars ={}".format(self.goalvar))
        
        self.proc_inputvars()
       #logger.info("Processed input vars ={}".format(','.join(self.input_vars)))
        
        if self.add_vars:
            self.proc_addvars()
        #   logger.info("Processed add vars ={}".format(','.join(self.add_vars)))
        #else:
        #   logger.info("No additional data provided")
        
        if split_randomly:
            self.split_data_randomly()
        elif not split_randomly:
            self.split_data_sequentially()
        
        logger.info("Fraction of Input/goal data for evaluation {}".format(self.eval_fraction))

        data_dict={}
        # PACK INPUT VARS
        data_dict['X_train']=self.X_train
        data_dict['X_eval'] = self.X_eval
        # PACK GOAL VARS
        data_dict['goalvar_train'] = self.goalvar_train
        data_dict['goalvar_eval'] = self.goalvar_eval

        return data_dict

    ###########################
    # REGRESSION TRAIN/PREDICT
    ##########################
    def regression(self,processed_data):
        # DESCRIPTION
        #       performing various kinds of regression algorithms on preprocessed data
        # INPUT
        #       max_depth_in   - maximim depth of a regression tree.
        #                        If None,nodes are expanded until :TODO ??
        #       processed_data - dictionary with 4 values containing
        #                        train/evaluation parts of input/goal variables
        #                        e.g. Masked array processed_data['X_train'] contains N=len(input_vars)
        #                        rows of shape (1- eval_fraction)nx*ny*150
        # OUTPUT
        #       goalvar_pred  -  ndarray of predicted(using regression) goal variable,
        #                        shape (eval_fraction*nx*ny*150)
        #       goalvar_eval  -  Masked array of goal variable read from netcdf, :TODO comparing differnt types of data OK?
        #                        shape (eval_fraction*nx*ny*150)

        # unpack input variables
        X_train=processed_data['X_train']
        X_eval = processed_data['X_eval']
        # unpack goal variable
        goalvar_train = processed_data['goalvar_train']
        goalvar_eval = processed_data['goalvar_eval']

        supported_types=['decision_tree','gradient_boost','random_forest']
        assert self.regtype in supported_types, 'entered regression type is not supported \n' \
                                           'supported types are: {}'.format(','.join(supported_types))

        # CHOOSE REGRESSION TYPE
        if self.regtype == 'decision_tree':
            logger.info("{} regression is chosen".format(self.regtype))
            regtree = tree.DecisionTreeRegressor(max_depth = self.max_depth)

        elif self.regtype == 'gradient_boost':
            logger.info("{} regression is chosen".format(self.regtype))
            regtree = ensemble.GradientBoostingRegressor(max_depth = self.max_depth)

        elif self.regtype == 'random_forest':
            logger.info("{} regression is chosen".format(self.regtype))
            if self.max_depth == None:
                regtree = ensemble.RandomForestRegressor()
            else:
                regtree = ensemble.RandomForestRegressor(n_estimators = self.max_depth,max_leaf_nodes = self.max_depth)

        a = datetime.datetime.now()

        # TRAINING
        logger.info("++++++++++Training begins++++++++++")
        regtree.fit(X_train, goalvar_train)
        # PREDICTION

        logger.info("++++++++++Prediction begins++++++++++")
        goalvar_pred = regtree.predict(X_eval)
        b = datetime.datetime.now()
        logger.debug(f'execution time = {b-a}')
        logger.debug("============================================================ \n")

        return goalvar_pred,goalvar_eval # TODO: should not return goalvar_eval, can be unpacked from processed_data dict



def hist_plot(goalvar_pred,goalvar_eval, eval_fraction,bins,vmax,cmax,norm):
    corrmat=np.corrcoef(goalvar_pred, goalvar_eval)
    corr=corrmat[0,1]

    fig,ax = plt.subplots(figsize=(10, 10))
    # colormap of lognormalized due to high values range.
    # empty regions correspond to log(0)=-inf
    im=ax.hist2d(goalvar_eval, goalvar_pred, bins=bins, norm=norm,cmap=plt.cm.jet, vmax=vmax,cmax=cmax)
    ax.set_box_aspect(1)
    ax.set_title(f'2D histogram ({bins} bins) of Cloud Fraction \n \
    			N elements={len(goalvar_eval)},fraction of data used for testing {eval_fraction} \n \
    			correlation coefficient is {corr:.4f}')
    ax.set_xlabel('predicted')
    ax.set_ylabel('pseudo-observations')
    cbar=fig.colorbar(im[3])
    cbar.ax.set_ylabel('                                  log(N), where N - \n                                        - number of (x,y) pairs/ 2d bin', rotation=0)
    plt.show()

    return None


def get_config_params(config_file_name='../config/config_variables.ini'):
    
    configParser = configparser.ConfigParser()
    configParser.read(config_file_name)
    filedir = configParser.get('INPUT','filedir')

    return filedir



def main():
    # DATA PREPROCESSING
    prepro=DataPrepro(abspath, vars_dict, eval_fraction, regtype, ML_max_depth,resolution = size)

    # methods should be in this particular order!!
    processed_data = prepro.get_processed_data()

    # REGRESSION
    goalvar_pred, goalvar_eval = regression(processed_data)

    # VISUALISATION
    hist_plot(goalvar_pred, goalvar_eval)

try   :
	# close previously opened handlers, if the cft is reimported
	# this is necessary to avoid multiplying logger outputs
	logger.handlers=[]
except:
	pass
logger = initLog()

if __name__=='__main__':

    # DEFAULT PARAMETERS - global variables
    #curdir='/home/igor/UNI/Master Project/Script/Data/'
    #curdir='/home/egordeev/002_Data'
    curdir = get_config_params()
    fname='ncr_pdf_douze_0125deg.nc'
    goal_var = 'cl_l'
    input_vars = ['qsm', 'qtm', 'qlm', 'skew_l', 'var_l', 'var_t', 'tm', 'pm']
    add_vars = ['qvlm','qsm']
    eval_fraction=0.6
    regtype = 'decision_tree'
    max_depth_in=None

    logger = initLog()
    logger.debug("EXECUTION START, input file={}".format(fname))

    main()

    logger.debug('EXECUTION FINISH %s seconds' % (start_time - end_time))
    logger.debug('----------------------------------------')

