# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:39:19 2024

@author: ryans
"""

import unittest
import pandas as pd
import matplotlib.pyplot as plt
from app_functions import df_converter 
from app_functions import compare 
from app_functions import who_to_start
from app_functions import full_graph
import numpy as np

class TestDfConverter(unittest.TestCase):
    def test_df_converter(self):
        # Arrange: Create a sample DataFrame (replace with your actual data)
        data = {'Name': ['Tom Brady', 'Drew Brees'], 'Position': ['QB', 'QB']}
        df = pd.DataFrame(data)

        # Act: Call the function
        result = df_converter(df)

        # Assert: Check if the result is a valid CSV encoded in utf-8
        self.assertIsInstance(result, bytes)
        

class TestCompare(unittest.TestCase):
    def test_compare(self):
        # synthetic test data - input players
        player_1 = 'Tom Brady'
        player_2 = 'Drew Brees'
        # player stats to be compared
        data = {
            'player_display_name': ['Tom Brady', 'Tom Brady', 'Tom Brady', 'Tom Brady',
                                    'Drew Brees', 'Drew Brees', 'Drew Brees', 'Drew Brees'],
            'week': [14, 15, 16, 17, 14, 15, 16, 17],
            'predicted': [17, 22, 26, 18, 33, 36, 19, 21],
        }
        # make it a dataframe - need for function
        df = pd.DataFrame(data)

        # call the function
        result = compare(player_1, player_2, df)

        # Assert: Check if the result is a valid matplotlib figure
        self.assertIsInstance(result, plt.Figure)
        
        
        
# who_to_start test. This will have two tests because we need to make sure 
class TestWhoToStart(unittest.TestCase):
    '''Test case for the who_to_start_function.'''
    def test_both_players_starting(self):
        # create synthetic data week input
        week = 14
        # create player input
        player_1 = 'Aaron Rodgers'
        player_2 = 'Matthew Stafford'
        # create dataframe
        data = {
            'player_display_name': ['Aaron Rodgers', 'Matthew Stafford'],
            'week': [14, 14],
            'predicted': [14, 22],
        }
        df = pd.DataFrame(data)
        
        # call the function
        result = who_to_start(week, player_1, player_2, df)
        # expected
        expected_result = f'Start {player_2}\nPlayer Predictions:\n{player_1}: [10]\n{player_2}: [12]'
        # assert equal
        self.assertEqual(result, expected_result)

    def test_neither_player_starting(self):
        week = 14
        player_1 = 'Player1'
        player_2 = 'Player2'
        data = {
            'player_display_name': ['OtherPlayer1', 'OtherPlayer2'],
            'week': [14, 14],
            'predicted': [8, np.nan],
        }
        df = pd.DataFrame(data)

        result = who_to_start(week, player_1, player_2, df)
        expected_result = f'Please Choose Two Players who are starting for week {week}.'
        self.assertEqual(result, expected_result)
        
class TestFullGraph(unittest.TestCase):
    def test_full_graph(self):
        # Create a sample DataFrame for testing
        data = {
            'player_display_name': ['Aaron Rodgers', 'Matthew Stafford'],
            'period': [1, 2],
            'fantasy_points_ppr': [10, 15],
            'predicted': [12, 18]
        }
        master_set = pd.DataFrame(data)

        # Call the function with a specific player
        player_name = 'Matthew Stafford'
        fig = full_graph(player_name, master_set)

        # Check if the figure is created
        self.assertIsNotNone(fig)


if __name__ == '__main__':
    unittest.main()