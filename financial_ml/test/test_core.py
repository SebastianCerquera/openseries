# +
import unittest
import pandas as pd

from financial_ml.core import TimeSerie, DF2TimeSerie, Differentitator


# -

class TimeSerieTest(unittest.TestCase):
    
    @staticmethod
    def read_dummy_df():
        return pd.read_parquet('/home/runner/notebooks/fiducias/data/dummy.parquet')

    def test_builder_df2timeseries(self):
        dummy_df = self.read_dummy_df()
        
        builder = DF2TimeSerie(pd.Timedelta('1d'))
        timeserie = builder.build(dummy_df)
        self.assertTrue(isinstance(timeserie, TimeSerie))
        
    def test_transformation_differentiator(self):
        dummy_df = self.read_dummy_df()
        
        builder = DF2TimeSerie(pd.Timedelta('1d'))
        timeserie = builder.build(dummy_df)
        
        differentiator = Differentitator()
        timeserie_diff = differentiator.transform(timeserie)
        
        self.assertTrue(isinstance(timeserie_diff, TimeSerie))
