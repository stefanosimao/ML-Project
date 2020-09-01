import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def run():
    
    #TRAIN DATA
    data = np.genfromtxt('train.csv', delimiter = ',')
    
    #REMOVE FIRST LINE
    data = data[1:, 1:] 

    #LAMBDAS
    lambda_parameters = [0.01, 0.1, 1, 10, 100]
    
    #FOLDS
    folds_number = 10
    
    #DEFINE RMSE
    rmse = np.zeros([folds_number, len(lambda_parameters)])
    
    #DIVIDE DATA WITH "n_splits = folds_number" MEANS LEAVING OUT A DIFFERENT FOLD EACH TIME
    k_fold = KFold(n_splits = folds_number, shuffle = False)
    
    for j in range(0, 5):
        i = 0
        for train, test in k_fold.split(data):
            
            #COMPUTE RIDGE REGRESSION
            regression = Ridge(alpha = lambda_parameters[j], fit_intercept= False, tol=1e-4)

            #DEFINE DATA TO USE
            X_train_data = data[train,1:]
            y_train_data = data[train,0]
            X_test_data = data[test,1:]
            y_test_data = data[test,0]
            
            #TRAIN MODELS
            regression = regression.fit(X_train_data,y_train_data)

            #PREDICT BASED ON TEST DATA
            y_predict_data = regression.predict(X_test_data)

            #COMPUTE RMSE
            rmse[i,j] = np.sqrt(mean_squared_error(y_predict_data, y_test_data))
            
            i += 1
                          
    #AVERAGE OF THE RMSEs
    mean_rmse = np.mean(rmse, 0)

    #SAVE RESULTS
    np.savetxt ('results_final.csv', mean_rmse, comments='')
    
    return

if __name__ == "__main__":
    run()