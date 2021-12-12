# -*- coding: utf-8 -*-
# +
import unittest

from financial_ml.io import CSV2TimeSerie, TimeSerie
from bancolombia.valores import ValoresBancolombiaCSV

# +
import os
import dotenv

dotenv.load_dotenv()
repo = os.environ.get('REPO')
# -


class CSV2TimeSerieTest(unittest.TestCase):
    
    def test_read_series_defaults(self):
        #give
        builder = CSV2TimeSerie(repo + '/data/raw.csv', "test serie")

        #when: Se construlle objeto Timeseria partiendo de un DataFrame
        timeserie = builder.build(None)

        #then
        self.assertTrue(isinstance(timeserie, TimeSerie))
        
    def test_read_with_non_default_names(self):
        #give
        builder = CSV2TimeSerie(
            repo + '/data/raw_non_default_column_names.csv',
            "test serie",
            value_column="unit_value",
            date_column="date"
        )

        #when: Se construlle objeto Timeseria partiendo de un DataFrame
        timeserie = builder.build(None)

        #then
        self.assertTrue(isinstance(timeserie, TimeSerie))
        
    def test_read_with_non_default_date_pattern(self):
        #give
        builder = CSV2TimeSerie(
            repo + '/data/raw_non_default_date_pattern.csv',
            "test serie",
            date_pattern="%Y/%m/%d"
        )

        #when: Se construlle objeto Timeseria partiendo de un DataFrame
        timeserie = builder.build(None)

        #then
        self.assertTrue(isinstance(timeserie, TimeSerie))
        
    def test_read_with_date_pattern_function(self):
        builder = ValoresBancolombiaCSV(
            repo + '/data/raw_non_default_date_pattern_function.csv',
            "test serie"
        )

        #when: Se construlle objeto Timeseria partiendo de un DataFrame
        timeserie = builder.build(None)

        #then
        self.assertTrue(isinstance(timeserie, TimeSerie))


unittest.main(argv=[''], verbosity=2, exit=False)


