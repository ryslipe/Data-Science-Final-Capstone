# -*- coding: utf-8 -*-
"""
Created on Mon May 20 19:17:48 2024

@author: ryans
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from PIL import Image

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import root_mean_squared_error
from streamlit_extras.no_default_selectbox import selectbox
import plotly.express as px
import app_functions as app
import pipeline_function as pipe
import plotly.graph_objects as go

st.set_page_config(layout='wide')


with st.sidebar:
    selected = option_menu(
        menu_title = 'Main Menu',
        options = ['Quarterbacks', 'Runningbacks', 'Wide Receivers', 'Tight Ends', 'User Guide'],
        default_index = 0
        )
# title of our app
st.title(':football: Fantasy Football Machine \tLearning Predictor :football:')
    
if selected == 'Quarterbacks':    

    # introductory paragraph
    st.write('Welcome to the Fantasy Football Machine Learning Predictor! In this first phase of rollouts, we are dealing with only quarterbacks. The data consists of training data fro the 2020, 2021, and first 13 weeks of the 2022 seasons. The model is then tested on the last 4 games of the 2022 season. Each season had the final game removed from the data because it is not representative of the population. In the final week of the season many teams rest their best players or play them in small amounts to avoid injury. We do not want this week to disturb the statistics used for prediction. The model uses a 12 weeek rolling average of various player statistics to come up with a prediction. For quarterbacks, a "lasso" model gave the lowest RMSE. It is tested on the last four weeks because this is generally the time frame of fantasy football playoff matchups.')
    
    # dataframes for qbs
    quarterbacks_full = pd.read_csv('data/quarterbacks_23_all_cols')
    df_qb = pd.read_csv('data/qb_final_df_23_new')
    qb_train = pd.read_csv('data/qb_training_23_rolling')
    df_table = df_qb.copy()
    df_table['season'] = df_table['season'].astype(str)

    ########################################################################################################################################################
    # first section - player predictions
    ########################################################################################################################################################
    st.header('Player Predictions - Final 4 Weeks of 2023')
    # explain the search bar
    st.write('To view the results of the model enter a player that you would like to see predictions for. If the player has no data it means they did not play during the final 4 games of the season. The sortable table includes the player name along with the week and actual and predicted points scored. Click on the column to sort the predictions.')
   
    # enter a player name to display predictions
    text_search = st.text_input('Enter a player name. If table is empty, player not found.', '')
    
    # call table creation function from app_functions
    table = app.make_table(text_search, df_qb)
    
    if text_search:
        searched_table = df_qb[table]
        searched_table['season'] = searched_table['season'].astype(str).str.replace(',', '')
        st.write(searched_table)
    
    # call app_function df_converter
    csv = app.df_converter(df_qb)

    # downloader
    st.write('\U0001F447 To see every quarterback''s predictions download the dataset here. \U0001F447')
    st.download_button(
     label="Download quarterback projections data as CSV",
     data=csv,
     file_name='qb_projections_df.csv',
     mime='text/csv',
 )
    
    ########################################################################################################################################################
    # second section week by week predictions
    ########################################################################################################################################################
    # write header
    st.header('Week by Week Predictions')
    
    # explain the week to week predictions
    st.write('Choose a week to display the predictions of every quarterback for the selected week.')
    
    # choose a week to display 
    text_2 = st.select_slider('Choose a Week Number', [14, 15, 16, 17])
    
    if text_2:
        df_qb['season'] = df_qb['season'].astype(str).str.replace(',', '')
        df_qb.loc[df_qb['week'] == text_2]

    ########################################################################################################################################################
    # third section - graphical comparison
    ########################################################################################################################################################
    st.header('Graphical Comparison')
    st.write('To make comparisons of two players easy to interpret, enter two players for a line graph of the predicted points for the final 4 games of the 2022 season. ')
    
    # input for player 1 and 2
    player_1 = st.text_input('Enter First Player', '').title()
    player_2 = st.text_input('Enter Second Player', '').title()
    
    if player_1 and player_2:
        fig = app.compare(player_1, player_2, df_qb)
        st.pyplot(fig)

    #########################################################################################################################################################
    # forth section - who to start
    #########################################################################################################################################################
    # write header
    st.header('Who to Start')
    
    # explain the "who to start" function
    st.write('Do you have two players that you are unsure about starting? These tough decisions could be costly. Let the model make the decision for you. Type in the week you want along with the two players you are deciding between and the model will tell you who you should start. If the player entered is not playing in those weeks you will be asked to try again.')  
    
    # input for player 1 and 2
    week_starter = st.selectbox('Pick a week for starting comparison', [14, 15, 16, 17])

    # create a select box - this dataset has players listed multiple times so use set()
    player = set(df_qb['player_display_name'])
    player_starter_1 = st.selectbox('Enter a player to start', player)
    player_starter_2 = st.selectbox('Enter a second player to start', player)
    
    if (week_starter) and (player_starter_1) and (player_starter_2):
        # call who to start function from app_functions.py
        app.who_to_start(int(week_starter), player_starter_1, player_starter_2, df_qb)

    ######################################################################################################################################################
    # section 5 - this section uses pipeline_function.py
    ######################################################################################################################################################
    # create X and y variables.
    X_train_qb = qb_train.drop(columns = ['player_id', 'player_display_name', 'fantasy_points_ppr'], axis = 1)
    y_train_qb = qb_train['fantasy_points_ppr'] 
    
    # call our function from pipeline_function.py to fit the models 
    qb_mods = pipe.model_creation(X_train_qb, y_train_qb)
    
    
    # call the function from pipeline_function.py
    qb_train_rmse = pipe.full_train_rmse(qb_mods, X_train_qb, y_train_qb)
    
    # call the plotting function
    ylim = [0, 9]
    fig_1 = pipe.make_rmse_plot(qb_train_rmse, 'RMSE Plot without Cross Validation', ylim)
    
    if st.button('Generate RMSE Report'):
        st.pyplot(fig_1)
        
    st.write('The results of the RMSE show that random forest is the best model but there is potenial for overfitting.')
    
    
    # random forest model shows us feature importances
    importances = pd.read_csv('data/importances.csv')
    st.write(importances)
    ######################################################################################################################################################
    # call the plotting function
    cv_rmse_dict = {'knn': 7.995205302511437,
     'rf': 7.920867882577945,
     'gb': 7.942460224774128,
     'ridge': 7.88654632047547,
     'lasso': 7.880494637687264}
    
    # graph the grid searched results 
    ylim = [7, 9]
    fig_2 = pipe.make_rmse_plot(cv_rmse_dict, 'Graph of Cross Validation RMSE', ylim)
    if st.button('Generate Grid Searched RMSE Report'):
        st.pyplot(fig_2)
        
    st.write('The results of the plot show the RMSE values got higher but not by too much. The lowest RMSE is from the Lasso model but they are all very close. This is the reason the model chosen was the Lasso model. In future rollouts, I will implement an ensemble of methods along with neural networks and time series analysis techniques.')
    
    
    st.header('Descriptive Statistics')
    st.write('The descriptive statistics are displayed below. Since the range of values are much different it is imoprtant to scale the data for the Lasso model.')
    st.write(qb_train.describe().T)
    
    st.write('One of the interesting parts of the data analysis is to look at the correlation of our features with our target variable. None of these are extremely correlated to the target alone, but with interactions among other variables, our predictions are quite accurate for most players. ')
    corr_matrix = qb_train.iloc[:, 2:].corr()
    st.write(corr_matrix['fantasy_points_ppr'].sort_values(ascending = False))

    ##########################################################################################################################################################
    # section 6 - projection overlay
    ##########################################################################################################################################################
    st.write('We can get a graph of our players actual points from the training data along with the projected points from the testing data to see how they are trending.')
    
    # set up for our full_graph function from app_functions.py
    # graph of the players training data along with testing data
    st.header('Projection Overlay')
    st.write('Choose a player from the drop down menu to see their historical points graphed in black and their projections graphed in red. If there is no red line it means the player did not play in the final four weeks of the 2023 season.')
    
    # players involved in analysis - must be involved in training data but not testing 
    player = set(qb_train['player_display_name'])

    # select box set up
    full_player = selectbox('Pick a player from the drop down menu.', player)

    # this is the player that is picked
    choice = full_player

    # this includes training data which is quarterbacks_full and testing data which is df_qb
    master_set = pd.concat([quarterbacks_full, df_qb], axis = 0, ignore_index = True)

    # create a period column for our dates
    master_set['period'] = master_set['season'].astype(str) + '.' + master_set['week'].astype(str)
        
    # call our full_graph function from app_functions.py
    if choice:
        fig3 = st.plotly_chart(app.full_graph(choice, master_set))

    ########################################################################################################################################################
    # section 7 - 2024 projections
    ########################################################################################################################################################
    image = Image.open('./pagebreak_img.jpg')
    st.image(image)
    # 2024 week 1 data
    st.header('2024 Week 1 Predictions')
    st.write('Now that the results of the model have been displayed, predictions on future games can be made. The predictions use the same model that made predictions on the last 4 games of the 2023 season but include all games from 2020-2024 week 1. These are the predictions for the first week of the 2024 season and will be updated every Tuesday to make predictions for the weeks that follow.')
    st.warning('WARNING: The depth charts for the 2024 season have not been finalized yet. This means there are backup players that may be projected to score points even though they will not be starters for that team. For example, Davis Mills is projected to score roughly 10 points although C.J. Stroud is their starting quarterback. Once the depth charts are finalized, this will be updated. For now, interpret it as: if Davis Mills starts week 1, he is projected to score roughly 10 points.', icon='⚠️')

    # 2024 quarterback dataframes
    quarterbacks_full_2024 = pd.read_csv('data/quarterbacks_24_all_cols')
    qb_train_2024 = pd.read_csv('data/qb_training_24_rolling')
    df_qb_2024 = pd.read_csv('data/qb_final_df_24.csv')

    # explain the search bar
    st.write('To view the results of the model select a player that you would like to see predictions for.')
   
    
    # enter a player name to display predictions
    player = set(df_qb_2024['player_display_name'])
    full_player = st.selectbox('Enter a player name. If table is empty, player not found.', player)
    
    player_choice = full_player
    columns_to_include = ['player_display_name', 'last_twelve_passing_yards', 'last_twelve_passing_tds', 'last_twelve_rushing_yards',
                          'last_twelve_rushing_tds', 'last_twelve_fantasy_points_ppr', 'predicted']
    
    if full_player:
        searched_table = df_qb_2024.loc[df_qb_2024['player_display_name'] == player_choice]
        searched_table = searched_table[columns_to_include]
        st.write(searched_table)

    # all quarterbacks in one table
    st.header('Predictions for all Quarterbacks')
    st.write('These are the predictions for every quarterback. To search a player, hover over the top right of the table and select the magnifying glass icon.')
    # only scoring columns
    qbs = df_qb_2024[columns_to_include]
    # display them
    st.write(qbs)

    #########################################################################################################################################################
    # who to start week 1 2024
    ##########################################################################################################################################################
    st.header('Who to Start - 2024 Week 1')
    
    # explain the "who to start" function
    st.write('Do you have two players that you are unsure about starting? These tough decisions could be costly. Let the model make the decision for you. Type in the two players you are deciding between and the model will tell you who you should start. If the player entered is not playing in those weeks you will be asked to try again.')  
    
    # input for player 1 and 2
    # create a select box 
    week_starter = 1
    player = df_qb_2024['player_display_name']
    player_starter_1 = st.selectbox('Enter a player to start', player)
    player_starter_2 = st.selectbox('Enter a second player to start', player)
    
    if (week_starter) and (player_starter_1) and (player_starter_2):
        # call who to start function from app_functions.py
        app.who_to_start(int(week_starter), player_starter_1, player_starter_2, df_qb_2024)
    
        
###############################################################################################################################################################    
###############################################################################################################################################################
###############################################################################################################################################################
    
if selected == 'Runningbacks':
    st.title(f'{selected} Coming Soon')
    # running back dataframes needed
    runningbacks_full = pd.read_csv('data/runningbacks_23_all_cols')
    df_rb = pd.read_csv('data/rb_final_df_23_new')
    rb_train = pd.read_csv('data/rb_training_23_rolling')
    df_table = df_rb.copy()
    df_table['season'] = df_table['season'].astype(str)
    
    # first section - player predictions
    st.header('Player Predictions')
    # explain the search bar
    st.write('To view the results of the model enter a player that you would like to see predictions for. If the player has no data it means they did not play during the final 4 games of the season. The sortable table includes the player name along with the week and actual and predicted points scored. Click on the column to sort the predictions.')
   
    
    # enter a player name to display predictions
    player = set(df_rb['player_display_name'])
    full_player = st.selectbox('Enter a player name. If table is empty, player not found.', player)
    player_choice = full_player
    
    if full_player:
        searched_table = df_rb.loc[df_rb['player_display_name'] == player_choice]
        searched_table['season'] = searched_table['season'].astype(str).str.replace(',', '')
        st.write(searched_table)

    # call app_function df_converter
    csv = app.df_converter(df_rb)

    # downloader
    st.write("\U0001F447 To see every runningback's predictions download the dataset here. \U0001F447")
    st.download_button(
     label="Download runningback projections data as CSV",
     data=csv,
     file_name='rb_projections_df.csv',
     mime='text/csv',
 )
    ########################################################################################################################################################
    # second section week by week predictions
    ########################################################################################################################################################
    # write header
    st.header('Week by Week Predictions')
    
    # explain the week to week predictions
    st.write('Choose a week to display the predictions of every quarterback for the selected week.')
    
    # choose a week to display 
    text_2 = st.select_slider('Choose a Week Number', [14, 15, 16, 17])
    
    if text_2:
        df_rb['season'] = df_rb['season'].astype(str).str.replace(',', '')
        df_rb.loc[df_rb['week'] == text_2]

    ########################################################################################################################################################
    # third section - graphical comparison
    ########################################################################################################################################################
    st.header('Graphical Comparison')
    st.write('To make comparisons of two players easy to interpret, enter two players for a line graph of the predicted points for the final 4 games of the 2022 season. ')
    
   # enter a player name to display predictions
    player = set(df_rb['player_display_name'])
    player_1 = st.selectbox('Choose a player from the list of players.', player)
    player_1_choice = player_1

    player_2 = st.selectbox('Choose another player from the list of players.', player)
    player_2_choice = player_2
    
    if player_1 and player_2:
        fig = app.compare(player_1_choice, player_2_choice, df_rb)
        st.pyplot(fig)
    ########################################################################################################################################################
    # fourth section - who to start
    ########################################################################################################################################################
    # write header
    st.header('Who to Start')
    
    # explain the "who to start" function
    st.write('Do you have two players that you are unsure about starting? These tough decisions could be costly. Let the model make the decision for you. Type in the week you want along with the two players you are deciding between and the model will tell you who you should start. If the player entered is not playing in those weeks you will be asked to try again.')  
    
    # input for player 1 and 2
    week_starter = st.selectbox('Pick a week for starting comparison', [14, 15, 16, 17])

    # create a select box - this dataset has players listed multiple times so use set()
    player = set(df_rb['player_display_name'])
    player_starter_1 = st.selectbox('Enter a runningback to start', player)
    player_starter_2 = st.selectbox('Enter a second runningback to start', player)
    
    if (week_starter) and (player_starter_1) and (player_starter_2):
        # call who to start function from app_functions.py
        app.who_to_start(int(week_starter), player_starter_1, player_starter_2, df_rb)
    ########################################################################################################################################################
    # feature importances
    ########################################################################################################################################################
    # speak about importances
    st.write('The models for each position showed similar results so there is no need to show how accurate they are with RMSE plots. One thing that may be useful though however is the feature importances from the random forest model for each position. Here are the feature importances for the running backs. This can help a team owner decide what stats they should be focusing on.')
    
    # random forest model feature importances for runningbacks
    importances_rb = pd.read_csv('data/rb_importances.csv')
    st.write(importances_rb)

    ########################################################################################################################################################
    # section 6 - projection overlay
    ########################################################################################################################################################
    # set up for our full_graph function from app_functions.py
    # graph of the players training data along with testing data
    st.header('Projection Overlay')
    st.write('Choose a player from the drop down menu to see their historical points graphed in black and their projections graphed in red. If there is no red line it means the player did not play in the final four weeks of the 2023 season.')
    
    # players involved in analysis - must be involved in training data but not testing 
    player = set(rb_train['player_display_name'])

    # select box set up
    full_player = selectbox('Pick a player from the drop down menu.', player)

    # this is the player that is picked
    choice = full_player

    # this includes training data which is quarterbacks_full and testing data which is df_qb
    master_set = pd.concat([runningbacks_full, df_rb], axis = 0, ignore_index = True)

    # create a period column for our dates
    master_set['period'] = master_set['season'].astype(str) + '.' + master_set['week'].astype(str)
        
    # call our full_graph function from app_functions.py
    if choice:
        fig3 = st.plotly_chart(app.full_graph(choice, master_set))












if selected == 'Wide Receivers':
    
    # wide receiver dataframes needed
    wide_receivers_full = pd.read_csv('data/wr_23_all_cols')
    df_wr = pd.read_csv('data/wr_final_df_23_new')
    wr_train = pd.read_csv('data/wr_training_23_rolling')
    df_table = df_wr.copy()
    df_table['season'] = df_table['season'].astype(str)

    # first section - player predictions
    st.header('Player Predictions')
    # explain the search bar
    st.write('To view the results of the model enter a player that you would like to see predictions for. If the player has no data it means they did not play during the final 4 games of the season. The sortable table includes the player name along with the week and actual and predicted points scored. Click on the column to sort the predictions.')
   
    # enter a player name to display predictions
    player = set(df_wr['player_display_name'])
    full_player = st.selectbox('Enter a player name. If table is empty, player not found.', player)
    player_choice = full_player
    
    if full_player:
        searched_table = df_wr.loc[df_wr['player_display_name'] == player_choice]
        searched_table['season'] = searched_table['season'].astype(str).str.replace(',', '')
        st.write(searched_table)

    # call app_function df_converter
    csv = app.df_converter(df_wr)

    # downloader
    st.write("\U0001F447 To see every wide receiver's predictions download the dataset here. \U0001F447")
    st.download_button(
     label="Download wide receiver projections data as CSV",
     data=csv,
     file_name='wr_projections_df.csv',
     mime='text/csv',
 )
    ########################################################################################################################################################
    # second section week by week predictions
    ########################################################################################################################################################
    # write header
    st.header('Week by Week Predictions')
    
    # explain the week to week predictions
    st.write('Choose a week to display the predictions of every wide receiver for the selected week.')
    
    # choose a week to display 
    text_2 = st.select_slider('Choose a Week Number', [14, 15, 16, 17])
    
    if text_2:
        df_wr['season'] = df_wr['season'].astype(str).str.replace(',', '')
        df_wr.loc[df_wr['week'] == text_2]

    ########################################################################################################################################################
    # third section - graphical comparison
    ########################################################################################################################################################
    st.header('Graphical Comparison')
    st.write('To make comparisons of two players easy to interpret, enter two players for a line graph of the predicted points for the final 4 games of the 2022 season. ')
    
    # enter a player name to display predictions
    player = set(df_wr['player_display_name'])
    player_1 = st.selectbox('Choose a player from the list of players.', player)
    player_1_choice = player_1

    player_2 = st.selectbox('Choose another player from the list of players.', player)
    player_2_choice = player_2
    
    if player_1 and player_2:
        fig = app.compare(player_1_choice, player_2_choice, df_wr)
        st.pyplot(fig)
    ########################################################################################################################################################
    # forth section - who to start
    ########################################################################################################################################################
    # write header
    st.header('Who to Start')
    
    # explain the "who to start" function
    st.write('Do you have two players that you are unsure about starting? These tough decisions could be costly. Let the model make the decision for you. Type in the week you want along with the two players you are deciding between and the model will tell you who you should start. If the player entered is not playing in those weeks you will be asked to try again.')  
    
    # input for player 1 and 2
    week_starter = st.selectbox('Pick a week for starting comparison', [14, 15, 16, 17])

    # create a select box - this dataset has players listed multiple times so use set()
    player = set(df_wr['player_display_name'])
    player_starter_1 = st.selectbox('Enter a wide receiver to start', player)
    player_starter_2 = st.selectbox('Enter a second wide receiver to start', player)
    
    if (week_starter) and (player_starter_1) and (player_starter_2):
        # call who to start function from app_functions.py
        app.who_to_start(int(week_starter), player_starter_1, player_starter_2, df_wr)
    ########################################################################################################################################################
    # feature importances
    ########################################################################################################################################################
    # speak about importances
    st.write('The models for each position showed similar results so there is no need to show how accurate they are with RMSE plots. One thing that may be useful though however is the feature importances from the random forest model for each position. Here are the feature importances for the wide receivers. This can help a team owner decide what stats they should be focusing on.')
    
    # random forest model feature importances for wr
    importances_wr = pd.read_csv('data/wr_importances.csv')
    st.write(importances_wr)


    # graph of the players training data along with testing data
    st.header('Projection Overlay')
    st.write('Choose a player from the drop down menu to see their historical points graphed in black and their projections graphed in red. If there is no red line it means the player did not play in the final four weeks of the 2023 season.')
    
    # players involved in analysis - must be involved in training data but not testing 
    player = set(wr_train['player_display_name'])

    # select box set up
    full_player = selectbox('Pick a player from the drop down menu.', player)

    # this is the player that is picked
    choice = full_player

    # this includes training data which is quarterbacks_full and testing data which is df_qb
    master_set = pd.concat([wide_receivers_full, df_wr], axis = 0, ignore_index = True)

    # create a period column for our dates
    master_set['period'] = master_set['season'].astype(str) + '.' + master_set['week'].astype(str)
        
    # call our full_graph function from app_functions.py
    if choice:
        fig3 = st.plotly_chart(app.full_graph(choice, master_set))





    
if selected == 'Tight Ends':

    # tight end dataframes needed
    tight_ends_full = pd.read_csv('data/te_23_all_cols')
    df_te = pd.read_csv('data/te_final_df_23_new')
    te_train = pd.read_csv('data/te_training_23_rolling')
    df_table = df_te.copy()
    df_table['season'] = df_table['season'].astype(str)

    # first section - player predictions
    st.header('Player Predictions')
    # explain the search bar
    st.write('To view the results of the model enter a player that you would like to see predictions for. If the player has no data it means they did not play during the final 4 games of the season. The sortable table includes the player name along with the week and actual and predicted points scored. Click on the column to sort the predictions.')
   
    
    # enter a player name to display predictions
    player = set(df_te['player_display_name'])
    full_player = st.selectbox('Enter a player name. If table is empty, player not found.', player)
    player_choice = full_player
    
    if full_player:
        searched_table = df_te.loc[df_te['player_display_name'] == player_choice]
        searched_table['season'] = searched_table['season'].astype(str).str.replace(',', '')
        st.write(searched_table)

    # call app_function df_converter
    csv = app.df_converter(df_te)

    # downloader
    st.write("\U0001F447 To see every tight end's predictions download the dataset here. \U0001F447")
    st.download_button(
     label="Download tight end projections data as CSV",
     data=csv,
     file_name='te_projections_df.csv',
     mime='text/csv',
 )
    ########################################################################################################################################################
    # second section week by week predictions
    ########################################################################################################################################################
    # write header
    st.header('Week by Week Predictions')
    
    # explain the week to week predictions
    st.write('Choose a week to display the predictions of every tight end for the selected week.')
    
    # choose a week to display 
    text_2 = st.select_slider('Choose a Week Number', [14, 15, 16, 17])
    
    if text_2:
        df_te['season'] = df_te['season'].astype(str).str.replace(',', '')
        df_te.loc[df_te['week'] == text_2]


    ########################################################################################################################################################
    # third section - graphical comparison
    ########################################################################################################################################################
    st.header('Graphical Comparison')
    st.write('To make comparisons of two players easy to interpret, enter two players for a line graph of the predicted points for the final 4 games of the 2022 season. ')
    
    # enter a player name to display predictions
    player = set(df_te['player_display_name'])
    player_1 = st.selectbox('Choose a player from the list of players.', player)
    player_1_choice = player_1

    player_2 = st.selectbox('Choose another player from the list of players.', player)
    player_2_choice = player_2
    
    if player_1 and player_2:
        fig = app.compare(player_1_choice, player_2_choice, df_te)
        st.pyplot(fig)

    # write header
    st.header('Who to Start')
    
    # explain the "who to start" function
    st.write('Do you have two players that you are unsure about starting? These tough decisions could be costly. Let the model make the decision for you. Type in the week you want along with the two players you are deciding between and the model will tell you who you should start. If the player entered is not playing in those weeks you will be asked to try again.')  
    
    # input for player 1 and 2
    week_starter = st.selectbox('Pick a week for starting comparison', [14, 15, 16, 17])

    # create a select box - this dataset has players listed multiple times so use set()
    player = set(df_te['player_display_name'])
    player_starter_1 = st.selectbox('Enter a tight end to start', player)
    player_starter_2 = st.selectbox('Enter a second tight end to start', player)
    
    if (week_starter) and (player_starter_1) and (player_starter_2):
        # call who to start function from app_functions.py
        app.who_to_start(int(week_starter), player_starter_1, player_starter_2, df_te)

    ########################################################################################################################################################
    # feature importances
    ########################################################################################################################################################
    # speak about importances
    st.write('The models for each position showed similar results so there is no need to show how accurate they are with RMSE plots. One thing that may be useful though however is the feature importances from the random forest model for each position. Here are the feature importances for the tight ends. This can help a team owner decide what stats they should be focusing on.')
    
    # random forest model feature importances for runningbacks
    importances_te = pd.read_csv('data/te_importances.csv')
    st.write(importances_te)

    ########################################################################################################################################################
    # projection overlay - tight ends
    ########################################################################################################################################################
    
    # graph of the players training data along with testing data
    st.header('Projection Overlay')
    st.write('Choose a player from the drop down menu to see their historical points graphed in black and their projections graphed in red. If there is no red line it means the player did not play in the final four weeks of the 2023 season.')
    
    # players involved in analysis - must be involved in training data but not testing 
    player = set(te_train['player_display_name'])

    # select box set up
    full_player = selectbox('Pick a player from the drop down menu.', player)

    # this is the player that is picked
    choice = full_player

    # this includes training data which is quarterbacks_full and testing data which is df_qb
    master_set = pd.concat([tight_ends_full, df_te], axis = 0, ignore_index = True)

    # create a period column for our dates
    master_set['period'] = master_set['season'].astype(str) + '.' + master_set['week'].astype(str)
        
    # call our full_graph function from app_functions.py
    if choice:
        fig3 = st.plotly_chart(app.full_graph(choice, master_set))

    
        
if selected == 'User Guide':
    st.title(f'{selected}')
    st.write('Welcome to the user guide for the Fantasy Football Machine Learning Predictor.')
    
    
