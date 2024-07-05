# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 18:28:23 2024

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
wr_train = df_train.loc[df_train['position'] == 'WR'].copy()

# Let's make a copy of the qb_train dataframe that still has the **player_display_name**, **season**, and **week** columns. We want to do this so that we can grab these columns after the predictions for the model have been made.
wr_23_all_cols = wr_train.copy()
wr_23_all_cols.to_csv('data/wr_23_all_cols', index = False)


# columns we are dropping
wr_dropping = ['position', 'recent_team', 'season',
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
pipe.drop_columns(wr_train, wr_dropping)

# drop na
wr_train.dropna(inplace = True)

# save to csv
wr_train.to_csv('data/wr_training_23_rolling', index = False)


# create X and y variables.
X_train_wr = wr_train.drop(columns = ['player_id', 'player_display_name', 'fantasy_points_ppr'], axis = 1)
y_train_wr = wr_train['fantasy_points_ppr'] 

# bring in pipeline
pipelines = pipe.pipelines

# call our function to fit the models 
wr_mods = pipe.model_creation(X_train_wr, y_train_wr)


# call the rmse function then print the results - no cv yet.
wr_train_rmse = pipe.full_train_rmse(wr_mods, X_train_wr, y_train_wr)

# check results
wr_train_rmse.items()

# call printing function with our dictionary
pipe.printing_rmse(wr_train_rmse)

# call the plotting function
title = 'RMSE Plot without Cross Validation'
ylim = [0, 8]
pipe.make_rmse_plot(wr_train_rmse, title, ylim)

###############################################################################
# Grid Search
###############################################################################
# bring in the grid from pipeline_function module
grid = pipe.grid

# call the function to initiate grid search
wr_mods_cv = pipe.grid_search_models(wr_mods, X_train_wr, y_train_wr)


# call function to create dictionary of all lowest RMSE of cross val models
wr_searched_rmse = pipe.cv_rmse(wr_mods_cv)

# call rmse printing function for neatly printed data.
pipe.printing_rmse(wr_searched_rmse)


# plot after CV - rf wins
title = 'RMSE Plot After Cross Validation'
ylim = [7.25, 7.42]
pipe.make_rmse_plot(wr_searched_rmse, title, ylim)

# random forest model shows us feature importances
feature_importances_wr = wr_mods_cv['rf'].best_estimator_._final_estimator.feature_importances_

# set up the features for feature importance
attributes = list(X_train_wr.columns)
# importance of our features
sorted(zip(feature_importances_wr, attributes), reverse = True)

# call the function for our importances
wr_importances = pipe.feature_importances(wr_mods_cv['rf'], X_train_wr)


# make importances a dataframe
wr_importances = pd.DataFrame(data = wr_importances, columns = ['Importance', 'Feature'])
wr_importances = wr_importances[['Feature', 'Importance']]
wr_importances.to_csv('data/wr_importances.csv', index = False)

# Testing Data
# create testing data
wr_test = df_test.loc[df_test['position'] == 'WR'].copy()

# keep name in a df
wr_test_full = wr_test.copy()

# call drop_qb_columns function for test data
pipe.drop_columns(wr_test, wr_dropping)

# drop na 
wr_test.dropna(inplace = True)

# these are the players we are testing on
wr_test_df = wr_test.to_csv('data/wr_23_test_new', index = False)

# create X and y variables.
X_test_wr = wr_test.drop(columns = ['player_id', 'player_display_name', 'fantasy_points_ppr'], axis = 1)
y_test_wr = wr_test['fantasy_points_ppr'] 

# call our function. The ridge model is the best model.         
pipe.prediction_rmse(wr_mods_cv, X_test_wr, y_test_wr)

# the best model is the ridge model
final_wr_model = wr_mods_cv['rf'].best_estimator_
final_wr_predictions = final_wr_model.predict(X_test_wr)


# this part is from HOML book, make sure to site it!
from scipy import stats
confidence = 0.95
squared_errors = (final_wr_predictions - y_test_wr) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                        loc = squared_errors.mean(),
                        scale = stats.sem(squared_errors)))

# Quarterbacks Dataframe with predictions and actual
ypred_wr = pd.Series(final_wr_predictions)
wr_name_final = wr_test_full['player_display_name'].tolist()
wr_week = wr_test_full['week'].tolist()
wr_points = wr_test_full['fantasy_points_ppr'].tolist()
wr_season = wr_test_full['season'].tolist()
wr_df_final = pd.DataFrame([wr_name_final, wr_week, wr_season, wr_points, ypred_wr]).T
wr_df_final.rename({0: 'player_display_name', 1: 'week', 2: 'season', 3:  'fantasy_points_ppr', 4: 'predicted'}, axis = 1, inplace = True)


# save to dataframe - this dataframe is the predictions of the last 4 weeks. (random forest model)
wr_df_final.to_csv('data/wr_final_df_23_new', index = False)
