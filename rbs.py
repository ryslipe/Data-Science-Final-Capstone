# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 11:11:35 2024

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
rb_train = df_train.loc[df_train['position'] == 'RB'].copy()

# Let's make a copy of the qb_train dataframe that still has the **player_display_name**, **season**, and **week** columns. We want to do this so that we can grab these columns after the predictions for the model have been made.
runningbacks_23_all_cols = rb_train.copy()
runningbacks_23_all_cols.to_csv('data/runningbacks_23_all_cols', index = False)


# columns we are dropping
rb_dropping = ['position', 'recent_team', 'season',
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
pipe.drop_columns(rb_train, rb_dropping)

# drop na
rb_train.dropna(inplace = True)

# save to csv
rb_train.to_csv('data/rb_training_23_rolling', index = False)


# create X and y variables.
X_train_rb = rb_train.drop(columns = ['player_id', 'player_display_name', 'fantasy_points_ppr'], axis = 1)
y_train_rb = rb_train['fantasy_points_ppr'] 

# bring in pipeline
pipelines = pipe.pipelines

# call our function to fit the models 
rb_mods = pipe.model_creation(X_train_rb, y_train_rb)


# call the rmse function then print the results - no cv yet.
rb_train_rmse = pipe.full_train_rmse(rb_mods, X_train_rb, y_train_rb)

# check results
rb_train_rmse.items()

# call printing function with our dictionary
pipe.printing_rmse(rb_train_rmse)

# call the plotting function
title = 'RMSE Plot without Cross Validation'
ylim = [0, 8]
pipe.make_rmse_plot(rb_train_rmse, title, ylim)

###############################################################################
# Grid Search
###############################################################################
# bring in the grid from pipeline_function module
grid = pipe.grid

# call the function to initiate grid search
rb_mods_cv = pipe.grid_search_models(rb_mods, grid, X_train_rb, y_train_rb)


# call function to create dictionary of all lowest RMSE of cross val models
rb_searched_rmse = pipe.cv_rmse(rb_mods_cv)

# call rmse printing function for neatly printed data.
pipe.printing_rmse(rb_searched_rmse)


# plot after CV - ridge wins
title = 'RMSE Plot After Cross Validation'
ylim = [7.5, 7.75]
pipe.make_rmse_plot(rb_searched_rmse, title, ylim)

# random forest model shows us feature importances
feature_importances_rb = rb_mods_cv['rf'].best_estimator_._final_estimator.feature_importances_

# set up the features for feature importance
attributes = list(X_train_rb.columns)
# importance of our features
sorted(zip(feature_importances_rb, attributes), reverse = True)

# call the function for our importances
rb_importances = pipe.feature_importances(rb_mods_cv['rf'], X_train_rb)


# make importances a dataframe
rb_importances = pd.DataFrame(data = rb_importances, columns = ['Importance', 'Feature'])
rb_importances = rb_importances[['Feature', 'Importance']]
rb_importances.to_csv('data/rb_importances.csv', index = False)

# Testing Data
# create testing data
rb_test = df_test.loc[df_test['position'] == 'RB'].copy()

# keep name in a df
rb_test_full = rb_test.copy()

# call drop_qb_columns function for test data
pipe.drop_columns(rb_test, rb_dropping)

# drop na 
rb_test.dropna(inplace = True)

# these are the players we are testing on
rb_test_df = rb_test.to_csv('data/rb_23_test_new', index = False)

# create X and y variables.
X_test_rb = rb_test.drop(columns = ['player_id', 'player_display_name', 'fantasy_points_ppr'], axis = 1)
y_test_rb = rb_test['fantasy_points_ppr'] 

# call our function. The ridge model is the best model.         
pipe.prediction_rmse(rb_mods_cv, X_test_rb, y_test_rb)

# the best model is the ridge model
final_rb_model = rb_mods_cv['ridge'].best_estimator_
final_rb_predictions = final_rb_model.predict(X_test_rb)


# this part is from HOML book, make sure to site it!
from scipy import stats
confidence = 0.95
squared_errors = (final_rb_predictions - y_test_rb) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                        loc = squared_errors.mean(),
                        scale = stats.sem(squared_errors)))

# Quarterbacks Dataframe with predictions and actual
ypred_rb = pd.Series(final_rb_predictions)
rb_name_final = rb_test_full['player_display_name'].tolist()
rb_week = rb_test_full['week'].tolist()
rb_points = rb_test_full['fantasy_points_ppr'].tolist()
rb_season = rb_test_full['season'].tolist()
rb_df_final = pd.DataFrame([rb_name_final, rb_week, rb_season, rb_points, ypred_rb]).T
rb_df_final.rename({0: 'player_display_name', 1: 'week', 2: 'season', 3:  'fantasy_points_ppr', 4: 'predicted'}, axis = 1, inplace = True)


# save to dataframe - this dataframe is the predictions of the last 4 weeks. (ridge model)
rb_df_final.to_csv('data/rb_final_df_23_new', index = False)
