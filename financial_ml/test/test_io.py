# -*- coding: utf-8 -*-
# +
import unittest

from financial_ml.io import CSV2TimeSerie, TimeSerie

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


unittest.main(argv=[''], verbosity=2, exit=False)

