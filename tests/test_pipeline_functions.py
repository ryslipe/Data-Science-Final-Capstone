# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 16:48:06 2024

@author: ryans
"""

'''Test cases for the pipeline functions.'''
from pipeline_function import drop_columns
from pipeline_function import model_creation
from pipeline_function import full_train_rmse
from pipeline_function import make_rmse_plot
from pipeline_function import grid_search_models
from pipeline_function import min_rmse
from pipeline_function import cv_rmse
from pipeline_function import feature_importances
from pipeline_function import prediction_rmse
from sklearn.datasets import make_classification
import pandas as pd
import unittest
import numpy as np





# test the drop_columns function
class TestDropColumns(unittest.TestCase):
    '''A test for our drop_columns function.'''
    def test_drop_columns(self):
        # create dataframe for dropping columns
        df = pd.DataFrame({
                'First': [1, 2, 3],
                'Second': ['A', 'B', 'C'],
                'Third' : ['X', 'Y', 'Z']
            })
        # use the df created
        col = ['First', 'Third']
        new = drop_columns(df, col)
        # columns remaining
        result = len(new.columns)
        # columns expected
        expected = 1
        
        # assert equal
        self.assertEqual(result, expected)



# create a small dataset to be tested
X = np.array(([0, 10], [5, 50]))
y = [20, 80]

class TestModelCreation(unittest.TestCase):
    # call function
    def test_model_creation(self):
        # fit to data
        result = model_creation(X, y)
        # test there are 5 models
        result_len = len(result.keys())
        # expected keys is 5, one for each model
        expected_keys = 5
        
        # assert equal
        self.assertEqual(result_len, expected_keys)
        
# rmse test case
# create model dictionary
X = np.array(([0, 10], [50, 75], [150, 200], [800, 320], [50, 0]))
y = [100, 125, 200, 300, 10]
mod_dict = model_creation(X, y)
class TestFullTrainRMSE(unittest.TestCase):
    '''Test the full_train_rmse funciton.'''
    def test_full_train_rmse(self):
        # call function
        result = full_train_rmse(mod_dict, X, y)
        # test result of rf model
        result_rmse = round(result['knn'], 2)
        # expected
        expected = 0.38
        
        # assertEqual
        self.assertEqual(result_rmse, expected)
        

# test the make_rmse_plot
class TestMakeRMSEPlot(unittest.TestCase):
    def test_make_rmse_plot(self):
    # Example RMSE dictionary - must have 5 for the colors used in function
        test_rmse_dict = {'knn': 2.0, 'rf': 1.0, 'gb': 1.5, 'ridge': 1.25, 'lasso': 0.75}
        # create title since we need it as parameter
        title = 'RMSE Comparison'
        # also need ylim
        ylim = (0, 2.5)
        
        # call the function
        make_rmse_plot(test_rmse_dict, title, ylim)

              
       
# test the grid search function
grid = {
    'knn': {
        'kneighborsregressor__n_neighbors': [2, 3],
    },
    'rf': {
        
        'randomforestregressor__max_features': [1, 2],
        
    },
    'gb':{
        'gradientboostingregressor__n_estimators': [5, 10],
        
    },
    'ridge':{
        'ridge__alpha': [20, 25],
    },
    'lasso': {
        'lasso__alpha': [0.25, 0.5]
    }
}

 
class TestGridSearchModels(unittest.TestCase):
    '''Unittest for grid_searched_mods'''
    def test_grid_searched_models(self):
        # create dummy data
        dummy_dict = mod_dict
        # X and y
        X = np.array(([0, 10], [50, 75], [150, 200], [800, 320], [50, 0]))
        y = [100, 125, 200, 300, 10]
        # create shortened grid
        
        # call the function
        result = grid_search_models(dummy_dict, grid, X, y)

        # Check if the keys in the result match the algo names
        algo = len(result.keys())
        expected = 5
        self.assertEqual(algo, expected)

# test min rmse
class TestMinRMSE(unittest.TestCase):
    '''Unit test for min_rmse().'''
    def test_min_rmse(self):
        # make rmse dataframe
       rmse = {'mean_test_score': [10, 5, 20, 15]} 
       # call function 
       result = min_rmse(rmse)
       # get min rmse of our test data
       expected = np.sqrt(5)
       
       # assert almost equal because of rounding
       self.assertAlmostEqual(result, expected, places=6)


# test cv_rmse
class TestCVRMSE(unittest.TestCase):
    '''Test the cv_rmse() function.'''
    def test_cv_rmse_non_empty_dict(self):
        # create dummy data
        dummy_dict = mod_dict
        # X and y
        X = np.array(([0, 10], [50, 75], [150, 200], [800, 320], [50, 0]))
        y = [100, 125, 200, 300, 10]
        
        # call grid search function
        first_result = grid_search_models(dummy_dict, grid, X, y)
        

        # Call the function
        result = cv_rmse(first_result)

        # Check if the keys in the result match the algo names
        algo = len(result.keys())
        expected = 5
        self.assertEqual(algo, expected)


    
# test feature importances
# create dummy data
dummy_dict = mod_dict
# X and y
X = np.array(([0, 10], [50, 75], [150, 200], [800, 320], [50, 0]))
y = [100, 125, 200, 300, 10]




# Generate synthetic dataset with 3 informative features
X, y = make_classification(
    n_samples=100,
    n_features=10,
    n_informative=3,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    random_state=0,
    shuffle=False,
)

X_df = pd.DataFrame(X)

# call the function
result = grid_search_models(dummy_dict, grid, X, y)
# set up rf model
rf_model = result['rf']
class TestFeatureImportances(unittest.TestCase):
    '''Test the feature importances model.'''
    def test_feature_importances(self):
        # use the dummy model
        dummy_model = rf_model
        # call the function
        results = feature_importances(dummy_model, X_df)
        # there should be 10 feature importances
        length = len(results)
        exp_len = 10
        # assert equal
        self.assertEqual(length, exp_len)
        

# test prediction rmse
class TestPredictionRMSE(unittest.TestCase):
    '''Unit test for the prediction_rmse() function.'''
    def test_prediction_rmse(self):
        # dummy grid searched models - needed as input
        searched = grid_search_models(dummy_dict, grid, X, y)
        # call the function
        actual = prediction_rmse(searched, X, y)
        # expected knn value
        expected = 0.35
        # actual
        actual_rmse = actual['knn']
        
        # assert equal
        self.assertEqual(expected, round(actual_rmse, 2))
        
        
        
    
    