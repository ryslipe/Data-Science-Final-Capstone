# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:21:08 2024

@author: ryans
"""

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import root_mean_squared_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# create our pipeline
pipelines = {
    'knn': make_pipeline(StandardScaler(), KNeighborsRegressor()),
    'rf' : make_pipeline(StandardScaler(), RandomForestRegressor()),
    'gb' : make_pipeline(StandardScaler(), GradientBoostingRegressor()),
    'ridge': make_pipeline(StandardScaler(), Ridge()),
    'lasso': make_pipeline(StandardScaler(), Lasso())
}

### FUNCTION 6
def drop_columns(df, columns_to_drop):
    '''Drop the columns we will not be using for our analysis.'''
    df.drop(columns = columns_to_drop, axis = 1, inplace = True)
    return df

# try to make that into a function.
def model_creation(X, y):
    dic = {}
    for algo, pipeline in pipelines.items():
        model = pipeline.fit(X, y)
        dic[algo] = model
    return dic

# function to see our RMSE on full training data (without cross validation being used.)
# rmse of our models on full training data - don't really need this tbh
def full_train_rmse(model_dict, X, y):
    '''A function to produce the RMSE on full training data without use of cross validation.'''
    rmse_models = {}
    for algo, model in model_dict.items():
        # make a prediction on training data using each model
        pred = model.predict(X)
        # calculate mse
        rmse = root_mean_squared_error(y, pred)
        # calculate rmse
        rmse_models[algo] = (rmse)
    return rmse_models

# neat way to print the results
def printing_rmse(rmse_dict):
    for k, v in rmse_dict.items():
        print(k,':',v)
        
# Graph the results (Make a better graph down the road but this works for now)
def make_rmse_plot(rmse_dict, title, ylim):
    x_val = ['knn', 'rf', 'gb', 'ridge', 'lasso']
    y_val = list(rmse_dict.values())
    # create the graph
    fig, ax = plt.subplots()
    ax.bar(x_val, y_val, color = ['Red', 'Green', 'Black', 'Orange', 'Blue'])
    ax.set_title(title, fontsize = 24)
    ax.set_ylabel('rmse', fontsize = 14)
    ax.set_ylim(ylim)
    plt.show()


#################################################################################
# Next, we get into the grid search. First, create the grid.
# grid 
grid = {
    'knn': {
        'kneighborsregressor__n_neighbors': [30, 35, 40, 45, 50],
    },
    'rf': {
        'randomforestregressor__n_estimators': [100, 150, 200],
        'randomforestregressor__max_features': [2, 3, 4, 5],
        'randomforestregressor__max_depth': [3, 4]
    },
    'gb':{
        'gradientboostingregressor__n_estimators': [100, 200, 300],
        'gradientboostingregressor__max_features': [2, 3, 4, 5],
        'gradientboostingregressor__learning_rate': [0.001, 0.01, 0.1, 0.5],
        'gradientboostingregressor__max_depth': [2, 3, 4]
    },
    'ridge':{
        'ridge__alpha': [20, 25, 30, 35],
    },
    'lasso': {
        'lasso__alpha': [0.25, 0.5, 0.75, 1]
    }
}

# fit the grid search for each model
def grid_search_models(position_model_dict, X, y):
    '''Function to perform gridsearch on all of our models.'''
    # empty dictionary to start
    searched_models = {}
    for algo, model in position_model_dict.items():
        # keep track of where its at
        print(f'Training the {algo} model...')
        # find best parameters
        search = GridSearchCV(model, grid[algo], scoring = 'neg_mean_squared_error', cv = 5, return_train_score= True)
        # fit those best parameters on training data
        search.fit(X, y)
        # add to dictionary
        searched_models[algo] = search
    return searched_models


### ANOTHER FUNCTION (NUMBER IT)
def min_rmse(results):
    '''Function to return the lowest RMSE score of each model.'''
    # set up list for each cv score
    scores = []
    # find the 'neg_mean_squared_error' for each cv
    for mean_score in results['mean_test_score']:
        # get rmse by taking sqrt of neg mean_score
            scores.append(np.sqrt(-mean_score))
    # return lowest rmse
    return min(scores)

### FUNCTION FOR ALL CV RMSE
def cv_rmse(searched_model_dict):
    '''A function to get the lowest RMSE of our models.'''
    all_rmse = {}
    for algo, score in searched_model_dict.items():
        result = min_rmse(searched_model_dict[algo].cv_results_)
        all_rmse[algo] = result
    return all_rmse


# make a function that gets feature importances and prints it 
def feature_importances(rf_model, X):
    '''Print the feature importances.'''
    feature_importances = rf_model.best_estimator_._final_estimator.feature_importances_
    # get columns to match feature importances
    attributes = list(X.columns)
    # importances
    importances = sorted(zip(feature_importances, attributes), reverse = True)
    return importances


# we already have our trained models from before. Now we just have to test them.
def prediction_rmse(searched_model_dict, X, y):
    '''Make Predictions on Testing Data and print RMSE for each algorithm in our pipeline.'''
    for algo, model in searched_model_dict.items():
        pred = model.predict(X)
        mse = mean_squared_error(y, pred)
        rmse = np.sqrt(mse)
        print(f'Metrics for {algo}: {rmse}')