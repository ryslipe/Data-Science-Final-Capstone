# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 20:07:36 2024

@author: ryans
"""
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go

#  functions used in our app
# function to create table
def make_table(text_search,df):
    '''A function to create player search of our final dataframe.'''
    table = df['player_display_name'].str.contains(text_search.title())
    return table

# function to download df as csv
# dataframe downloader
@st.cache_data
def df_converter(df):
    '''Function to generate csv downloader.'''
    return df.to_csv().encode('utf-8')

# function to make graph of comparisons
def compare(player_1, player_2, df):
    '''A function to graph comparision of two players.'''
    # first player line graph
    first_line = df.loc[df['player_display_name'] == player_1]
    # second player line graph
    second_line = df.loc[df['player_display_name'] == player_2]
    
    # graph them
    fig, ax = plt.subplots(figsize = (10, 6))
    # x is the week y is predicted points
    ax.plot(first_line['week'], first_line['predicted'], label = player_1, marker = 'o')
    # x is the week y is predicted points
    ax.plot(second_line['week'], second_line['predicted'], label = player_2, marker = 'o')
    # week numbers
    plt.xticks([14, 15, 16, 17])
    # add title
    plt.title(f"Comparison of {player_1} and {player_2}")
    # add x label
    plt.xlabel('Week')
    # add y label
    plt.ylabel('Fantasy Points')
    plt.grid(True)
    # add legend
    plt.legend()
    return fig

# function for who to start section
def who_to_start(week, player_1, player_2, df):
    '''A function to decide which player should start.'''
    # subset of dataframe that is the player and week number
    player_1_name = df.loc[(df['player_display_name'] == player_1) & (df['week'] == week)]
    # change predictions to a list
    player_1_points = player_1_name['predicted'].tolist()
    
    # player 2 subset that is name and week number given
    player_2_name = df.loc[(df['player_display_name'] == player_2) & (df['week'] == week)]
    # change predictions to list for this player
    player_2_points = player_2_name['predicted'].tolist()
    
    # if both players have predictions for given week...
    if player_1_points and player_2_points:
    
        # names
        names = [player_1, player_2]
        
        # points
        points = [player_1_points, player_2_points]
        
        # get max points predicted
        most_points = max(points)
        
        # who to start
        starter = points.index(most_points)
        
        # best player is the name of the player with max points
        best_player = names[starter]

        # center the results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(' ')
        with col2:
            # write the results
            st.write(f'Start: {best_player}')
            st.write('Player Predictions:')
            st.write(f'{player_1}: {player_1_points[0]}')
            st.write(f'{player_2}: {player_2_points[1]}')
        with col3:
            st.write(' ')
        
        
    # if both players are not starting in that week let the user know.
    else:
        st.write(f'Please Choose Two Players who are starting for week {week}.')
        
        
# function for projection overlay section
def full_graph(player, master_set):
    '''Function to graph a player's actual from training and projected from testing.'''
    # Filter data for the specified player
    actual = master_set.loc[master_set['player_display_name'] == player]
    actual.reset_index(inplace=True)
    actual['index'] = actual.index

    # Extract projected values
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
