# +
import unittest
import pandas as pd

from financial_ml.agg_features import *
from financial_ml.core import TimeSerie, DF2TimeSerie

# +
import os
import dotenv

dotenv.load_dotenv()
repo = os.environ.get('REPO')


class FeaturesExtractorTest(unittest.TestCase):
    
    @staticmethod
    def sample_timeseries(size=None):
        dummy_df = pd.read_parquet(repo + '/data/dummy.parquet')
        
        if size is not None:
            dummy_df = dummy_df.iloc[:size]
        
        builder = DF2TimeSerie(pd.Timedelta('1d'))
        return builder.build(dummy_df)

    def test_weekly_aggregation(self):
        ## given:
        timeserie = self.sample_timeseries()
        
        ## when: Se transforma la serie en una nueva de agregados
        extractor = WeeklyExtractor('last_week_sum', sum)
        random_variable = extractor.transform(timeserie)
        
        ## then:
        self.assertTrue(isinstance(random_variable, ContinousVariable))
        self.assertTrue(random_variable.to_df().shape[0], timeserie.to_df().shape[0] - 7)

    def test_failing_aggregation(self):
        ## given:
        timeserie = self.sample_timeseries(5)
        
        ## when: Se transforma la serie en una nueva de agregados
        extractor = WeeklyExtractor('last_week_sum', sum)
        
        ### I really don't like this sintax, it is workaround to keep the test clean
        ### https://ongspxm.gitlab.io/blog/2016/11/assertraises-testing-for-errors-in-unittest/
        random_variable = lambda : extractor.transform(timeserie)
        
        ## then:
        self.assertRaises(Exception, random_variable)


# -
unittest.main(argv=[''], verbosity=2, exit=False)


