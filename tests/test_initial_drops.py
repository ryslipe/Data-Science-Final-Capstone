# -*- coding: utf-8 -*-
"""
Created on Sat May 18 15:26:25 2024

@author: ryans
"""

import unittest
from data_preprocessing import columns_to_drop
from data_preprocessing import initial_drops
from data_preprocessing import df_new
from data_preprocessing import rolling_avg_try
from data_preprocessing import statistic_avgs_try
from data_preprocessing import shifting
import pandas as pd
import nfl_data_py as nfl

# create a dataframe to be tested
test_df = nfl.import_weekly_data(range(2019, 2020))

# completion percentage and usage
# create completion percentage and usage 
test_df['comp_percentage'] = test_df['completions'] / test_df['attempts']
test_df['usage'] = test_df['carries'] + test_df['targets']


class TestInitialDrop(unittest.TestCase):
    '''Tests for initial_drops function.'''
    def test_initial_drops(self):
        result_df = initial_drops(test_df)
        
        self.assertEqual(list(result_df.columns), list(df_new.columns)) 
        
if __name__ == '__main__':
    unittest.main()
    
# new dataframe to be tested for rolling_avg_try()
test_rolling = test_df.iloc[14:22, :]

# use groupby then apply lambda 
class TestRollingAvgTry(unittest.TestCase):
    '''Test the rolling averages using groupby id.'''
    def test_rolling_avg_try(self):
        # get rolling averages for two players
        test_rolling['rolling'] = test_df.groupby('player_id')['completions'].apply(lambda x: rolling_avg_try(x, 2)).reset_index(0, drop = True)
        # combine drew brees with the expected sum
        test_brees = test_rolling.loc[test_rolling['player_display_name'] == 'Drew Brees']
        # sum of rolling averages
        rolling_sum = test_brees['rolling'].sum()
        # expected sum
        expected_sum = 32 + 17.5 + 18.5 + 33 + 30
        
        # assert equal
        self.assertEqual(rolling_sum, expected_sum)
   
if __name__ == '__main__':
    unittest.main()
    

# test the statistic avg function on a small dataset
class TestStatisticAvgsTry(unittest.TestCase):
    '''Test the statistic rolling averages on a column.'''
    def test_statistic_avgs_try(self):
        # just these two columns 
        col = ['passing_yards', 'passing_tds']
        # set up the new column by calling the function
        avg_try = statistic_avgs_try(test_rolling, col)
        # get brees twelve week rolling td
        brees_rolling = avg_try.loc[avg_try['player_display_name'] == 'Drew Brees']
        # tds
        brees_td = round(brees_rolling['last_twelve_passing_tds'].sum(), 2)
        # expected total for brees
        expected_td = 7.52
        
        # assert equal
        self.assertEqual(brees_td, expected_td)

if __name__ == '__main__':
    unittest.main()        
    
    
# test shifting function on small dataset
class TestShiftingFunction(unittest.TestCase):
    '''Test the shifting function for NA values (should be present in 1st row per player per last twelve).'''
    def test_shifting(self):
        # shift passing_yards and passing_tds
        col = ['last_twelve_passing_yards', 'last_twelve_passing_tds']
        # get avgs
        avg_try = statistic_avgs_try(test_rolling, col)
        # shift the two columns 
        shifted = shifting(avg_try, col)
        # count null values for each column
        null_yards = shifted['last_twelve_passing_yards'].isnull().sum()
        null_tds = shifted['last_twelve_passing_tds'].isnull().sum()
        # total nulls
        total = null_yards + null_tds
        # expected is 4
        expected = 4
        
        # assert equal
        self.assertEqual(total, expected)
        

        
        