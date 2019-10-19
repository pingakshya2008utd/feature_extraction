import pandas as pd
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
from sklearn.linear_model import LinearRegression
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error


def main():
    opt = Optimizer("0FyTyofhjonvvEMrssqMng6pC")
    experiment = Experiment(api_key="0FyTyofhjonvvEMrssqMng6pC",
                        project_name="route_experiment_horizontal", workspace="pingakshya2008")


    dataframe = pd.read_csv("C:/ddr_read/ISPD4/A3.csv")
    #dataframe = pd.read_csv('C:/ddr_read/ISPD4/ispd2_ispd4.csv')
    print(dataframe.head(5))

    array = dataframe.values
# separate array into input and output components
    X = array[:,0:9]
    Y = array[:,10]


    dataframe_test = pd.read_csv("C:/ddr_read/ISPD2/ispd2_final_horizontal.csv")
    print(dataframe_test.head(5))

    array_test = dataframe_test.values
    # separate array into input and output components
    X_test_test = array_test[:,0:10]
    Y_test_test = array_test[:,11]


#Y = dataframe[' Hori route cong (%)']
    print("----------------------xxxx-------------------")
    print(X)

    print("----------------------yyyy-------------------")
    print(Y)
#scaler = MinMaxScaler(feature_range=(0, 1))
#rescaledX = scaler.fit_transform(X)
# summarize transformed data

    scaler = MinMaxScaler(feature_range=(-1, 1))
    rescaledX = scaler.fit_transform(X)
    rescaledX_test = scaler.fit_transform(X_test_test)
    numpy.set_printoptions(precision=4)
    #print(rescaledX[3:9,:])
    validation_size = 0.30
    seed = 10
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(rescaledX, Y, test_size=validation_size, random_state=seed)
    print("--- x train-------------")
    print(X_train)

    '''
    # pcs for MLP
    pcs_content = """hidden_layer_sizes integer [1000,2000] [1500]
    solver categorical {sgd,adam,lbfgs} [adam]
    activation categorical {identity,logistic,tanh,relu} [relu]
    learning_rate categorical {constant,invscaling,adaptive} [constant]  
    """
    '''


    i=0


    ###pcs for random forest
    pcs_content="""n_estimators integer [10,100] [11]
    min_samples_split integer [2,20] [3] 
    min_samples_leaf real [0,0.499] [0.1]
    max_features categorical {auto,sqrt,log2,None} [auto]
    max_leaf_nodes integer [50,150] [100]
    bootstrap categorical {True,False} [True] 
    """

    '''
    ### pcs for Linear Regression
    pcs_content="""fit_intercept categorical {True,False} [True]
    normalize categorical {True,False} [False]
    copy_X categorical {True,False} [True]
    """
    '''

    opt.set_params(pcs_content)
    while True:
        i=i+1
        try:
            sug = opt.get_suggestion()
        except NoMoreSuggestionsAvailable:
            break
        print("SUG", sug, sug.__dict__)

        #if i==700 :
        #    break

        '''
        ##/ ** ** ** ** ** ** ** ** estimators for Linear Regression **** ** ** ** ** ** ** ** * /
        fi=sug["fit_intercept"]
        no= normalize=["normalize"]
        cx=sug["copy_X"]

        print("fit_intercept= ",repr(fi),"normalize= ",repr(no),"copy_X= ",repr(cx))

        clf= LinearRegression(fit_intercept=sug["fit_intercept"], normalize=sug["normalize"], copy_X=sug["copy_X"])

        '''
        '''
       # /**************** estimators for MLP *******************/
        flu = sug["hidden_layer_sizes"]
        sol=sug["solver"]
        print("FLU", repr(flu))
        print("sol", repr(sol))
        clf = MLPRegressor(hidden_layer_sizes=sug["hidden_layer_sizes"], solver=sug["solver"],activation=sug["activation"],
                           alpha=0.0001, batch_size='auto', learning_rate=sug["learning_rate"], learning_rate_init=0.001,
                           power_t=0.05, max_iter=200, shuffle=True, random_state=10, tol=0.0001, verbose=False,
                           warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False,
                           validation_fraction=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        
        '''
        

        # /**************** estimators for Random Forrest Rgressor *******************/

        ne=sug["n_estimators"]
        ms=sug["min_samples_split"]
        ml=sug["min_samples_leaf"]
        mln=sug["max_leaf_nodes"]
        bs=sug["bootstrap"]
        oob="false"

        print("estimator= ",repr(ne),"mean sample split= ", repr(ms),"min sample leaf= ", repr(ml),"max leaf nodes= ", repr(mln),"bootstrap= ", repr(bs),
              "oob=", repr(oob), "i= ", i)


        clf = sklearn.ensemble.RandomForestRegressor(n_estimators=sug["n_estimators"], criterion='mse', max_depth=10,
                                                     min_samples_split=sug["min_samples_split"], min_samples_leaf=sug["min_samples_leaf"],
                                                     max_features='auto',
                                                     max_leaf_nodes=sug["max_leaf_nodes"], bootstrap=sug["bootstrap"], oob_score=False, n_jobs=1,
                                                     random_state=10, verbose=0)



        '''
             activation='relu', solver='adam',
                           alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001,
                           power_t=0.05, max_iter=200, shuffle=True, random_state=10, tol=0.0001, verbose=False,
                           warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False,
                           validation_fraction=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            
            
        '''

        clf.fit(X_train, Y_train)
        y_predict = clf.predict(X_validation)
        print('R2 score validation %.5f' % r2_score(Y_validation, y_predict))
        score= r2_score(Y_validation, y_predict)
        sug.report_score("accuracy", score)




if __name__ == "__main__":
    main()
