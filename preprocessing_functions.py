# -*- coding: utf-8 -*-
"""
Created on Thu May 30 14:45:21 2024

@author: ryans
"""
import pandas as pd

# drop these columns
columns_to_drop = ['player_name', 'position_group', 'headshot_url', 'season_type',
       'sacks', 'sack_yards',
       'sack_fumbles', 'passing_air_yards',
       'passing_yards_after_catch', 'passing_epa',
       'passing_2pt_conversions', 'pacr', 'dakota', 'rushing_fumbles', 
       'rushing_epa', 'rushing_2pt_conversions',
       'receiving_fumbles', 'receiving_air_yards', 'receiving_epa',
       'receiving_2pt_conversions', 'racr', 'air_yards_share',
       'wopr', 'special_teams_tds', 'fantasy_points']




## FUNCTION NUMBER 1 
def initial_drops(df):
    '''Drop columns and rows we will not be examining.'''
    # not using week 18
    relevant = df['week'] < 18
    
    # boolean indexing
    df = df[relevant]
    
    # establish columns to be dropped
    columns = columns_to_drop
    df.drop(columns = columns, axis = 1, inplace = True)
    
    return df


### FUNCTION NUMBER 2
def rolling_avg_try(df, window):
    '''A function to compute the rolling averages of our statistics.'''
    # shift by one so we do not include this weeks unknown statistics
    return df.rolling(window = window, min_periods = 1).mean()

# stats that we want rolling averages for 
avgs = ['completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions', 'sack_fumbles_lost', 'passing_first_downs', 'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles_lost', 'rushing_first_downs', 'receptions', 'targets', 'receiving_yards', 'receiving_tds', 'receiving_fumbles_lost', 'receiving_yards_after_catch', 'receiving_first_downs', 'target_share', 'fantasy_points_ppr', 'usage', 'comp_percentage']


### FUNCTION NUMBER 3
def statistic_avgs_try(df, col):
    '''Get rolling averages for specific columns.'''
    for col in df[col]:
        df[f'last_twelve_{col}'] = df.groupby('player_id')[col].apply(lambda x: rolling_avg_try(x, 12)).reset_index(0, drop = True)
    return df

# columns to be shifted 
last_twelve = ['last_twelve_completions', 'last_twelve_attempts',
'last_twelve_passing_yards', 'last_twelve_passing_tds',
'last_twelve_interceptions', 'last_twelve_sack_fumbles_lost',
'last_twelve_passing_first_downs', 'last_twelve_carries',
'last_twelve_rushing_yards', 'last_twelve_rushing_tds',
'last_twelve_rushing_fumbles_lost', 'last_twelve_rushing_first_downs',
'last_twelve_receptions', 'last_twelve_targets',
'last_twelve_receiving_yards', 'last_twelve_receiving_tds',
'last_twelve_receiving_fumbles_lost',
'last_twelve_receiving_yards_after_catch',
'last_twelve_receiving_first_downs', 'last_twelve_target_share',
'last_twelve_fantasy_points_ppr', 'last_twelve_usage',
'last_twelve_comp_percentage', 'rolling_def']

# function to shift them 
def shifting(df, col):
    for col in df[col]:
        df[col] = df.groupby('player_id')[col].shift(1)
    return df




