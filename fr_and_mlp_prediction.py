import pandas as pd
import time
from comet_ml import Experiment,  Optimizer, NoMoreSuggestionsAvailable
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import sklearn
from sklearn import model_selection
from pyearth import Earth
from sklearn.svm import SVR
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import scipy
import numpy
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error
#import sklearn.neural_network.MLPRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error

 

 

file = open("C:/ddr_read/ISPD4/example_results_horizontal_mae.csv","w")
file.write("design name, MAE\n")

for x in range(9, 10):

   

  

    name = 'C:/ddr_read/ISPD4/fpga' +str(x)+ '.csv'

    dataframe = pd.read_csv(name)

    array = dataframe.values

# separate array into input and output components

    X = array[:,3:13]

    Y = array[:,14]

 

    scaler = MinMaxScaler(feature_range=(-1, 1))

    rescaledX = scaler.fit_transform(X)

 

    validation_size = 0.35

    seed = 10

    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(rescaledX, Y, test_size=validation_size, random_state=seed)

    start_time = time.time()
    


    clf = sklearn.ensemble.RandomForestRegressor(n_estimators=97, criterion='mse', max_depth=10,

                                                     min_samples_split=10, min_samples_leaf=0.0002835320,

                                                     max_features='auto',

                                                     max_leaf_nodes=149, bootstrap=True, oob_score=False, n_jobs=1,

                                                     random_state=10, verbose=False)
    '''
    clf = MLPRegressor(hidden_layer_sizes=1761, solver='adam', activation='relu',
                       alpha=0.0001, batch_size='auto', learning_rate='adaptive', learning_rate_init=0.001,
                       power_t=0.05, max_iter=200, shuffle=True, random_state=10, tol=0.0001, verbose=False,
                       warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False,
                       validation_fraction=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    

 

 

     '''

    clf.fit(X_train, Y_train)
    y_predict = clf.predict(X_validation)
    stop_time = time.time()
    delay = stop_time - start_time
    score=r2_score(Y_validation, y_predict)
    mae=mean_absolute_error(Y_validation, y_predict)
   # mpe=numpy.mean(numpy.abs((Y_validation - y_predict) / Y_validation))
    print('fpga', x, 'R2 score validation %.5f' % r2_score(Y_validation, y_predict), 'Time= ', delay, 'MAE= ',mae)

    buf = 'fpga' + str(x) + ','+str(mae)+'\n'

    file.write(buf)

 

file.close()
