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
    
    # first section - player predictions
    st.header('Player Predictions')
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
    
    st.write('\U0001F447 To see every quarterback''s predictions download the dataset here. \U0001F447')
    st.download_button(
     label="Download quarterback projections data as CSV",
     data=csv,
     file_name='qb_projections_df.csv',
     mime='text/csv',
 )
    
    
    st.header('Week by Week Predictions')
    st.write('Choose a week to display the predictions of every quarterback for the selected week.')
    # choose a week to display 
    text_2 = st.select_slider('Choose a Week Number', [14, 15, 16, 17])
    
    if text_2:
        df_qb['season'] = df_qb['season'].astype(str).str.replace(',', '')
        df_qb.loc[df_qb['week'] == text_2]
    
    # next section - graphical comparison
    st.header('Graphical Comparison')
    st.write('To make comparisons of two players easy to interpret, enter two players for a line graph of the predicted points for the final 4 games of the 2022 season. ')
    
    # input for player 1 and 2
    player_1 = st.text_input('Enter First Player', '').title()
    player_2 = st.text_input('Enter Second Player', '').title()
    
    if player_1 and player_2:
        fig = app.compare(player_1, player_2, df_qb)
        st.pyplot(fig)
        
    # call app_functions who_to_start
    # next section - who to start
    st.header('Who to Start')  
    # explain the "who to start" function
    st.write('Do you have two players that you are unsure about starting? These tough decisions could be costly. Let the model make the decision for you. Type in the week you want along with the two players you are deciding between and the model will tell you who you should start. If the player entered is not playing in those weeks you will be asked to try again.')  
    # input for player 1 and 2
    week_starter = st.selectbox('Pick a week for starting comparison', [14, 15, 16, 17])
    player = set(qb_train['player_display_name'])
    player_starter_1 = st.selectbox('Enter a player to start', player)
    player_starter_2 = st.selectbox('Enter a second player to start', player)
    
    if (week_starter) and (player_starter_1) and (player_starter_2):
    
        app.who_to_start(int(week_starter), player_starter_1, player_starter_2, df_qb)

    
    # create X and y variables.
    X_train_qb = qb_train.drop(columns = ['player_id', 'player_display_name', 'fantasy_points_ppr'], axis = 1)
    y_train_qb = qb_train['fantasy_points_ppr'] 
    
    ######################################################################################################################################################
    # create our pipeline
    pipelines = {
        'knn': make_pipeline(StandardScaler(), KNeighborsRegressor()),
        'rf' : make_pipeline(StandardScaler(), RandomForestRegressor()),
        'gb' : make_pipeline(StandardScaler(), GradientBoostingRegressor()),
        'ridge': make_pipeline(StandardScaler(), Ridge()),
        'lasso': make_pipeline(StandardScaler(), Lasso())
    }
    
    # these will be our 5 models. It will fit to our training data and create a dictionary of each model. 
    # try to make that into a function.
    def model_creation(X, y):
        dic = {}
        for algo, pipeline in pipelines.items():
            model = pipeline.fit(X, y)
            dic[algo] = model
        return dic
    
    
    # call our function to fit the models 
    qb_mods = model_creation(X_train_qb, y_train_qb)
    
    
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
    
    # call the function then print the results.
    qb_train_rmse = full_train_rmse(qb_mods, X_train_qb, y_train_qb)
    qb_train_rmse.items()
    
    
    # bar graph of our rmse results
    # set up x and y values
    x_val = ['knn', 'rf', 'gb', 'ridge', 'lasso']
    y_val = list(qb_train_rmse.values())
    
    
    # Graph the results 
    def make_rmse_plot(rmse_dict, title, ylim):
        x_val = ['knn', 'rf', 'gb', 'ridge', 'lasso']
        y_val = list(rmse_dict.values())
        # create the graph
        fig_1, ax = plt.subplots()
        ax.bar(x_val, y_val, color = ['Red', 'Green', 'Black', 'Orange', 'Blue'])
        ax.set_title(title, fontsize = 24)
        ax.set_ylabel('rmse', fontsize = 14)
        ax.set_ylim(ylim)
        return fig_1
    
    # call the plotting function
    ylim = [0, 9]
    fig_1 = make_rmse_plot(qb_train_rmse, 'RMSE Plot without Cross Validation', ylim)
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
    fig_2 = make_rmse_plot(cv_rmse_dict, 'Graph of Cross Validation RMSE', ylim)
    if st.button('Generate Grid Searched RMSE Report'):
        st.pyplot(fig_2)
        
    st.write('The results of the plot show the RMSE values got higher but not by too much. The lowest RMSE is from the Lasso model but they are all very close. This is the reason the model chosen was the Lasso model. In future rollouts, I will implement an ensemble of methods along with neural networks and time series analysis techniques.')
    
    
    st.header('Descriptive Statistics')
    st.write('The descriptive statistics are displayed below. Since the range of values are much different it is imoprtant to scale the data for the Lasso model.')
    st.write(qb_train.describe().T)
    
    st.write('One of the interesting parts of the data analysis is to look at the correlation of our features with our target variable. None of these are extremely correlated to the target alone, but with interactions among other variables, our predictions are quite accurate for most players. ')
    corr_matrix = qb_train.iloc[:, 2:].corr()
    st.write(corr_matrix['fantasy_points_ppr'].sort_values(ascending = False))
    
    st.write('We can get a graph of our players actual points from the training data along with the projected points from the testing data to see how they are trending.')
    
    
    
    # graph of the players training data along with testing data
    player = set(qb_train['player_display_name'])
    st.header('Projection Overlay')
    st.write('Choose a player from the drop down menu to see their historical points graphed in black and their projections graphed in red. If there is no red line it means the player did not play in the final four weeks of the 2023 season.')
    full_player = selectbox('Pick a player from the drop down menu.', player)
    choice = full_player
    master_set = pd.concat([quarterbacks_full, df_qb], axis = 0, ignore_index = True)

    master_set['period'] = master_set['season'].astype(str) + '.' + master_set['week'].astype(str)
    
    # take season 2024 out because we do not need it in this analysis
    
    actual = master_set.loc[master_set['player_display_name'] == player]

    
    import plotly.graph_objects as go
    
    def full_graph(player, master_set):
        '''Function to graph a player's actual from training and projected from testing.'''
        # Filter data for the specified player
        actual = master_set.loc[master_set['player_display_name'] == player]
        actual.reset_index(inplace=True)
        actual['index'] = actual.index
    
        # Extract actual and projected values
        y_vals = actual['fantasy_points_ppr']
        test_projections = actual['predicted']
    
        # Create a Plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=actual['period'], y=actual['fantasy_points_ppr'],
                                 mode='lines+markers',
                                 name='Actual Points'))
        
        fig.add_trace(go.Scatter(x=actual['period'], y=test_projections,
                                 mode='lines+markers',
                                 name='Projected'))
        fig.update_xaxes(rangeslider_visible = True)
        # Customize the figure
        fig.update_layout(title=f"{player}'s Fantasy Points",
                          title_font_size = 24,
                          xaxis_title ='Period',
                          yaxis_title = 'Fantasy Points',
                          template="plotly_dark",
                          width = 1000, height = 600)
        return fig
        

    
    if choice:
        fig3 = st.plotly_chart(full_graph(choice, master_set))
        
        
    
    
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
    
    if text_search:
        searched_table = df_rb.loc[df_rb['player_display_name'] == player_choice]
        searched_table['season'] = searched_table['season'].astype(str).str.replace(',', '')
        st.write(searched_table)
        
if selected == 'Wide Receivers':
    st.title(f'{selected} Coming Soon')
if selected == 'Tight Ends':
    st.title(f'{selected} Coming Soon')
if selected == 'User Guide':
    st.title(f'{selected}')
    st.write('Welcome to the user guide for the Fantasy Football Machine Learning Predictor.')
    
    
