# +
import unittest
import pandas as pd

from financial_ml.core import TimeSerie, DF2TimeSerie, Differentitator

# +
import os
import dotenv

dotenv.load_dotenv()
repo = os.environ.get('REPO')


# -

class TimeSerieTest(unittest.TestCase):
    
    @staticmethod
    def read_dummy_df():
        return pd.read_parquet(repo + '/data/dummy.parquet')

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
        
    def test_to_df(self):
        #given
        dummy_df = self.read_dummy_df()
        builder = DF2TimeSerie(pd.Timedelta('1d'))
        
        #when
        timeserie = builder.build(dummy_df)
        df = timeserie.to_df()
        
        #then
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertEqual(dummy_df.shape[0],df.shape[0])
        
    def test_apply(self):
        #given
        dummy_df = self.read_dummy_df()
        builder = DF2TimeSerie(pd.Timedelta('1d'))
        timeserie = builder.build(dummy_df)
        callback = lambda df:df
        
        #when
        timeSerie_traforme = timeserie.apply(callback)
        df = timeSerie_traforme.to_df()
        
        #then
        self.assertTrue(isinstance(timeSerie_traforme,TimeSerie))
        self.assertEqual(dummy_df.shape[0],df.shape[0])


unittest.main(argv=[''], verbosity=2, exit=False)


