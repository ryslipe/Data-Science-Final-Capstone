# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 15:37:18 2024

@author: ryans
"""

# this is the data for 2024. We can do all positions in the same document (I think)
# First up, dependencies

import pipeline_function as pipe
import numpy as np
import pandas as pd

# Next up, import training and testing data for all positions.
full_training = pd.read_csv('data/full_training.csv')

# testing - this includes all positions and has 55 columns. Send through the pipeline.
full_testing = pd.read_csv('data/full_testing.csv')

# get data for each position. I think it may be easier to just do all positions separately but at the same time one after another instead of starting the pipeline over from the beginning.
qb_train_2024 = full_training.loc[full_training['position'] == 'QB'].copy()
rb_train_2024 = full_training.loc[full_training['position'] == 'RB'].copy()
wr_train_2024 = full_training.loc[full_training['position'] == 'WR'].copy()
te_train_2024 = full_training.loc[full_training['position'] == 'TE'].copy()

# get the testing data for 2024 predictions
qb_test_2024 = full_testing.loc[full_testing['position'] == 'QB'].copy()
rb_test_2024 = full_testing.loc[full_testing['position'] == 'RB'].copy()
wr_test_2024 = full_testing.loc[full_testing['position'] == 'WR'].copy()
te_test_2024 = full_testing.loc[full_testing['position'] == 'TE'].copy()

# Let's make a copy of the qb_train dataframe that still has the **player_display_name**, **season**, and **week** columns. We want to do this so that we can grab these columns after the predictions for the model have been made.
quarterbacks_24_all_cols = qb_train_2024.copy()
quarterbacks_24_all_cols.to_csv('data/quarterbacks_24_all_cols', index = False)

runningbacks_24_all_cols = rb_train_2024.copy()
runningbacks_24_all_cols.to_csv('data/runningbacks_24_all_cols', index = False)

widereceivers_24_all_cols = wr_train_2024.copy()
widereceivers_24_all_cols.to_csv('data/widereceivers_24_all_cols', index = False)

tightends_24_all_cols = te_train_2024.copy()
tightends_24_all_cols.to_csv('data/tightends_24_all_cols', index = False)

# columns that we are dropping for each position. WR and TE are the same 
# drop these columns
qb_dropping = ['position', 'recent_team', 'season',
       'week', 'opponent_team', 'completions', 'attempts', 'passing_yards',
       'passing_tds', 'interceptions', 'sack_fumbles_lost',
       'passing_first_downs', 'carries', 'rushing_yards', 'rushing_tds',
       'rushing_fumbles_lost', 'rushing_first_downs', 'receptions', 'targets',
       'receiving_yards', 'receiving_tds', 'receiving_fumbles_lost',
       'receiving_yards_after_catch', 'receiving_first_downs', 'target_share',
       'usage', 'comp_percentage', 
       'last_twelve_receptions',
       'last_twelve_targets', 'last_twelve_receiving_yards',
       'last_twelve_receiving_tds', 'last_twelve_receiving_fumbles_lost',
       'last_twelve_receiving_yards_after_catch',
       'last_twelve_receiving_first_downs', 'last_twelve_target_share', 'last_twelve_usage', 'def_fantasy_points']

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


# drop columns for each position. Use pos_train variable.
# qb
pipe.drop_columns(qb_train_2024, qb_dropping)
# rb
pipe.drop_columns(rb_train_2024, rb_dropping)
# wr
pipe.drop_columns(wr_train_2024, wr_dropping)
# te
pipe.drop_columns(te_train_2024, te_dropping)


# drop na for all pos
# qb
qb_train_2024.dropna(inplace = True)
# rb
rb_train_2024.dropna(inplace = True)
# wr
wr_train_2024.dropna(inplace = True)
# te
te_train_2024.dropna(inplace = True)

# save these to a csv. These now include the rolling averages. This is useful for the projection overlay.
# save to csv
qb_train_2024.to_csv('data/qb_training_24_rolling', index = False)
rb_train_2024.to_csv('data/rb_training_24_rolling', index = False)
wr_train_2024.to_csv('data/wr_training_24_rolling', index = False)
te_train_2024.to_csv('data/te_training_24_rolling', index = False)


# create X an y for all positions
# qb
# create X and y variables.
X_train_qb_2024 = qb_train_2024.drop(columns = ['player_id', 'player_display_name', 'fantasy_points_ppr'], axis = 1)
y_train_qb_2024 = qb_train_2024['fantasy_points_ppr'] 

# rb
# create X and y variables.
X_train_rb_2024 = rb_train_2024.drop(columns = ['player_id', 'player_display_name', 'fantasy_points_ppr'], axis = 1)
y_train_rb_2024 = rb_train_2024['fantasy_points_ppr'] 

# wr
# create X and y variables.
X_train_wr_2024 = wr_train_2024.drop(columns = ['player_id', 'player_display_name', 'fantasy_points_ppr'], axis = 1)
y_train_wr_2024 = wr_train_2024['fantasy_points_ppr'] 

# te
# create X and y variables.
X_train_te_2024 = te_train_2024.drop(columns = ['player_id', 'player_display_name', 'fantasy_points_ppr'], axis = 1)
y_train_te_2024 = te_train_2024['fantasy_points_ppr'] 



# Now we start on our pipeline. The pipeline is the same for each position. Import it from the pipeline_functions module
pipelines = pipe.pipelines


# call the modle_creation function to create the models for all positions
qb_mods_24 = pipe.model_creation(X_train_qb_2024, y_train_qb_2024)
rb_mods_24 = pipe.model_creation(X_train_rb_2024, y_train_rb_2024)
wr_mods_24 = pipe.model_creation(X_train_wr_2024, y_train_wr_2024)
te_mods_24 = pipe.model_creation(X_train_te_2024, y_train_te_2024)

# call rmse function on the training data before cross validation
qb_train_2024_rmse = pipe.full_train_rmse(qb_mods_24, X_train_qb_2024, y_train_qb_2024)
rb_train_2024_rmse = pipe.full_train_rmse(rb_mods_24, X_train_rb_2024, y_train_rb_2024)
wr_train_2024_rmse = pipe.full_train_rmse(wr_mods_24, X_train_wr_2024, y_train_wr_2024)
te_train_2024_rmse = pipe.full_train_rmse(te_mods_24, X_train_te_2024, y_train_te_2024)

# call plotting function - qbs
title = 'RMSE without Cross Validation'
ylim = [2.5, 9]
pipe.make_rmse_plot(qb_train_2024_rmse, title, ylim)

# grid search for each position.
# qb
qb_mods_cv_2024 = pipe.grid_search_models(qb_mods_24, pipe.grid, X_train_qb_2024, y_train_qb_2024)

# rb
rb_mods_cv_2024 = pipe.grid_search_models(rb_mods_24, pipe.grid, X_train_rb_2024, y_train_rb_2024)

# wr
wr_mods_cv_2024 = pipe.grid_search_models(wr_mods_24, pipe.grid, X_train_wr_2024, y_train_wr_2024)

# te
te_mods_cv_2024 = pipe.grid_search_models(te_mods_24, pipe.grid, X_train_te_2024, y_train_te_2024)



# use cv_rmse to generate best model
# qbs
qb_searched_mods = pipe.cv_rmse(qb_mods_cv_2024) # best model is lasso

# rbs
rb_searched_mods = pipe.cv_rmse(rb_mods_cv_2024) # best model is ridge

# wrs
wr_searched_mods = pipe.cv_rmse(wr_mods_cv_2024) # best model is rf

# te
te_searched_mods = pipe.cv_rmse(te_mods_cv_2024) # best model is rf


# best model - qb
final_qb_model_2024 = qb_mods_cv_2024['lasso']
final_qb_model_2024 = final_qb_model_2024.best_estimator_

# best model - rb
final_rb_model_2024 = rb_mods_cv_2024['ridge']
final_rb_model_2024 = final_rb_model_2024.best_estimator_

# best model - wr
final_wr_model_2024 = wr_mods_cv_2024['rf']
final_wr_model_2024 = final_wr_model_2024.best_estimator_

# best model - te
final_te_model_2024 = te_mods_cv_2024['rf']
final_te_model_2024 = final_te_model_2024.best_estimator_

# Now set up the testing data for the models.
# drop the columns for each position
# qbs
pipe.drop_columns(qb_test_2024, qb_dropping)
# rbs
pipe.drop_columns(rb_test_2024, rb_dropping)
# wrs
pipe.drop_columns(wr_test_2024, wr_dropping)
# te
pipe.drop_columns(te_test_2024, te_dropping)

# qb set up
# drop fantasy_points_ppr because this has not happened yet
qb_test_2024.drop(columns = 'fantasy_points_ppr', axis = 1, inplace = True)

# must drop NaN values.
qb_test_2024.dropna(inplace = True)

# reset index
qb_test_2024.reset_index(drop = True, inplace = True)


# rb set up
# drop fantasy_points_ppr because this has not happened yet
rb_test_2024.drop(columns = 'fantasy_points_ppr', axis = 1, inplace = True)

# must drop NaN values.
rb_test_2024.dropna(inplace = True)

# reset index
rb_test_2024.reset_index(drop = True, inplace = True)


# wr set up
# drop fantasy_points_ppr because this has not happened yet
wr_test_2024.drop(columns = 'fantasy_points_ppr', axis = 1, inplace = True)

# must drop NaN values.
wr_test_2024.dropna(inplace = True)

# reset index
wr_test_2024.reset_index(drop = True, inplace = True)


# te set up
# drop fantasy_points_ppr because this has not happened yet
te_test_2024.drop(columns = 'fantasy_points_ppr', axis = 1, inplace = True)

# must drop NaN values.
te_test_2024.dropna(inplace = True)

# reset index
te_test_2024.reset_index(drop = True, inplace = True)


# make predictions using X_test
# X_test for each position
# qb
X_test_qb_2024 = qb_test_2024.drop(columns = ['player_id', 'player_display_name'], axis = 1)

# predictions 
qb_pred_2024 = final_qb_model_2024.predict(X_test_qb_2024)
qb_pred_series = pd.Series(qb_pred_2024)

# merge results with qb_test_2024
qb_test_2024 = qb_test_2024.merge(qb_pred_series.to_frame(), left_index = True, right_index = True)

# rename the last column
qb_test_2024.rename(columns = {0 : 'predicted'}, inplace = True )

# add season and week columns
qb_test_2024['season'] = 2024
qb_test_2024['week'] = 1


# rb
X_test_rb_2024 = rb_test_2024.drop(columns = ['player_id', 'player_display_name'], axis = 1)

# predictions 
rb_pred_2024 = final_rb_model_2024.predict(X_test_rb_2024)
rb_pred_series = pd.Series(rb_pred_2024)

# merge results with qb_test_2024
rb_test_2024 = rb_test_2024.merge(rb_pred_series.to_frame(), left_index = True, right_index = True)

# rename the last column
rb_test_2024.rename(columns = {0 : 'predicted'}, inplace = True )

# add season and week columns
rb_test_2024['season'] = 2024
rb_test_2024['week'] = 1


# wr
X_test_wr_2024 = wr_test_2024.drop(columns = ['player_id', 'player_display_name'], axis = 1)

# predictions 
wr_pred_2024 = final_wr_model_2024.predict(X_test_wr_2024)
wr_pred_series = pd.Series(wr_pred_2024)

# merge results with qb_test_2024
wr_test_2024 = wr_test_2024.merge(wr_pred_series.to_frame(), left_index = True, right_index = True)

# rename the last column
wr_test_2024.rename(columns = {0 : 'predicted'}, inplace = True )

# add season and week columns
wr_test_2024['season'] = 2024
wr_test_2024['week'] = 1


# te
X_test_te_2024 = te_test_2024.drop(columns = ['player_id', 'player_display_name'], axis = 1)

# predictions 
te_pred_2024 = final_te_model_2024.predict(X_test_te_2024)
te_pred_series = pd.Series(te_pred_2024)

# merge results with qb_test_2024
te_test_2024 = te_test_2024.merge(te_pred_series.to_frame(), left_index = True, right_index = True)

# rename the last column
te_test_2024.rename(columns = {0 : 'predicted'}, inplace = True )

# add season and week columns
te_test_2024['season'] = 2024
te_test_2024['week'] = 1

# save each to csv
qb_test_2024.to_csv('data/qb_final_df_24.csv')
rb_test_2024.to_csv('data/rb_final_df_24.csv')
wr_test_2024.to_csv('data/wr_final_df_24.csv')
te_test_2024.to_csv('data/te_final_df_24.csv')
