# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 19:02:07 2024

@author: ryans
"""

import pipeline_function as pipe
import numpy as np
import pandas as pd

# import training data
df_train = pd.read_csv('data/all_pos_train_23_new.csv')

# import testing data
df_test = pd.read_csv('data/all_pos_test_23_new.csv')

# get data for runningbacks- all 55 columns no 2024 data
te_train = df_train.loc[df_train['position'] == 'TE'].copy()

# Let's make a copy of the qb_train dataframe that still has the **player_display_name**, **season**, and **week** columns. We want to do this so that we can grab these columns after the predictions for the model have been made.
te_23_all_cols = te_train.copy()
te_23_all_cols.to_csv('data/te_23_all_cols', index = False)


# columns we are dropping
te_dropping = ['position', 'recent_team', 'season',
       'week', 'opponent_team', 'completions', 'attempts', 'passing_yards',
       'passing_tds', 'interceptions', 'sack_fumbles_lost',
       'passing_first_downs', 'carries', 'rushing_yards', 'rushing_tds',
       'rushing_fumbles_lost', 'rushing_first_downs', 'receptions', 'targets',
       'receiving_yards', 'receiving_tds', 'receiving_fumbles_lost',
       'receiving_yards_after_catch', 'receiving_first_downs', 'target_share', 'usage', 'comp_percentage', 
       'last_twelve_completions',
       'last_twelve_attempts', 'last_twelve_passing_yards', 'last_twelve_passing_tds', 
       'last_twelve_interceptions', 'last_twelve_sack_fumbles_lost', 'last_twelve_passing_first_downs','last_twelve_comp_percentage', 'def_fantasy_points']

# drop columns 
pipe.drop_columns(te_train, te_dropping)

# drop na
te_train.dropna(inplace = True)

# save to csv
te_train.to_csv('data/te_training_23_rolling', index = False)


# create X and y variables.
X_train_te = te_train.drop(columns = ['player_id', 'player_display_name', 'fantasy_points_ppr'], axis = 1)
y_train_te = te_train['fantasy_points_ppr'] 

# bring in pipeline
pipelines = pipe.pipelines

# call our function to fit the models 
te_mods = pipe.model_creation(X_train_te, y_train_te)

# call the rmse function then print the results - no cv yet.
te_train_rmse = pipe.full_train_rmse(te_mods, X_train_te, y_train_te)

# check results
te_train_rmse.items()

# call printing function with our dictionary
pipe.printing_rmse(te_train_rmse)

# call the plotting function
title = 'RMSE Plot without Cross Validation'
ylim = [0, 8]
pipe.make_rmse_plot(te_train_rmse, title, ylim)

###############################################################################
# Grid Search
###############################################################################
# bring in the grid from pipeline_function module
grid = pipe.grid

# call the function to initiate grid search
te_mods_cv = pipe.grid_search_models(te_mods, grid, X_train_te, y_train_te)


# call function to create dictionary of all lowest RMSE of cross val models
te_searched_rmse = pipe.cv_rmse(te_mods_cv)

# call rmse printing function for neatly printed data.
pipe.printing_rmse(te_searched_rmse)


# plot after CV - rf wins
title = 'RMSE Plot After Cross Validation'
ylim = [6, 6.2]
pipe.make_rmse_plot(te_searched_rmse, title, ylim)

# random forest model shows us feature importances
feature_importances_te = te_mods_cv['rf'].best_estimator_._final_estimator.feature_importances_

# set up the features for feature importance
attributes = list(X_train_te.columns)
# importance of our features
sorted(zip(feature_importances_te, attributes), reverse = True)

# call the function for our importances
te_importances = pipe.feature_importances(te_mods_cv['rf'], X_train_te)


# make importances a dataframe
te_importances = pd.DataFrame(data = te_importances, columns = ['Importance', 'Feature'])
te_importances = te_importances[['Feature', 'Importance']]
te_importances.to_csv('data/te_importances.csv', index = False)

# Testing Data
# create testing data
te_test = df_test.loc[df_test['position'] == 'TE'].copy()

# keep name in a df
te_test_full = te_test.copy()

# call drop_qb_columns function for test data
pipe.drop_columns(te_test, te_dropping)

# drop na 
te_test.dropna(inplace = True)

# these are the players we are testing on
te_test_df = te_test.to_csv('data/te_23_test_new', index = False)

# create X and y variables.
X_test_te = te_test.drop(columns = ['player_id', 'player_display_name', 'fantasy_points_ppr'], axis = 1)
y_test_te = te_test['fantasy_points_ppr'] 

# call our function. The lasso model is the best model.         
pipe.prediction_rmse(te_mods_cv, X_test_te, y_test_te)

# the best model is the ridge model
final_te_model = te_mods_cv['lasso'].best_estimator_
final_te_predictions = final_te_model.predict(X_test_te)


# this part is from HOML book, make sure to site it!
from scipy import stats
confidence = 0.95
squared_errors = (final_te_predictions - y_test_te) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                        loc = squared_errors.mean(),
                        scale = stats.sem(squared_errors)))

# Quarterbacks Dataframe with predictions and actual
ypred_te = pd.Series(final_te_predictions)
te_name_final = te_test_full['player_display_name'].tolist()
te_week = te_test_full['week'].tolist()
te_points = te_test_full['fantasy_points_ppr'].tolist()
te_season = te_test_full['season'].tolist()
te_df_final = pd.DataFrame([te_name_final, te_week, te_season, te_points, ypred_te]).T
te_df_final.rename({0: 'player_display_name', 1: 'week', 2: 'season', 3:  'fantasy_points_ppr', 4: 'predicted'}, axis = 1, inplace = True)


# save to dataframe - this dataframe is the predictions of the last 4 weeks. (lasso model)
te_df_final.to_csv('data/te_final_df_23_new', index = False)
