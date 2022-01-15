import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree,ensemble
# TEST sklearn on noise sine

# Create a random dataset
#rng = np.random.seed(14)
# time series parameters
nt=1000
WN_std=0.1


WN_arr=np.random.normal(loc=0,scale=WN_std,size=nt)
#X=np.linspace(0,4*np.pi,1000)
X=np.sort(4*np.pi * np.random.rand(nt, 1), axis=0)
sine=np.sin(X.flatten())
noisy_sine=sine+WN_arr
# split array into 2 parts - training and testing
Y=noisy_sine

def split_dataset_randomly(test_fraction,Y,X):
    random_indices=np.random.permutation(np.arange(nt))
    test_fraction=0.2
    training_fraction=1-test_fraction
    random_test_indices=random_indices[:int(nt*test_fraction)]
    random_train_indices=random_indices[int(nt*test_fraction):]
    Y_train=Y[random_train_indices]
    X_train=X[random_train_indices].reshape(len(X_train),1)

    Y_test=Y[random_test_indices]
    X_test=X[random_test_indices].reshape(len(X_test),1)

    return Y_train, Y_test, X_train, X_test

# plot two subsets for training and testing
#plt.scatter(X_train,Y_train,s=7); plt.scatter(X_test,Y_test,s=7); plt.show()
#fit regression model
regr1=tree.DecisionTreeRegressor(max_depth=2)
regr2=tree.DecisionTreeRegressor(max_depth=5)

regr1.fit(X,Y)
regr2.fit(X,Y)


X_test=np.linspace(0,4*np.pi,nt).reshape(nt,1)
# predict
Y1=regr1.predict(X_test)
Y2=regr2.predict(X_test)

# estimate the results
#plt.hist2d(Y2,Y_test);plt.show()
plt.plot(X_test,Y2,label='max_depth=5',linewidth=2)
plt.plot(X_test,Y1,label='max_depth=2',linewidth=2)
plt.scatter(X,Y,s=6,color='red',label='data')
plt.legend()
plt.show()



