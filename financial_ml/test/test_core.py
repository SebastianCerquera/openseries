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
        #give
        dummy_df = self.read_dummy_df()
        builder = DF2TimeSerie(pd.Timedelta('1d'))

        #when: Se construlle objeto Timeseria partiendo de un DataFrame
        timeserie = builder.build(dummy_df)

        #then
        self.assertTrue(isinstance(timeserie, TimeSerie))
        
    def test_transformation_differentiator(self):
        #given
        dummy_df = self.read_dummy_df()
        builder = DF2TimeSerie(pd.Timedelta('1d'))
        timeserie = builder.build(dummy_df)
        
        #when: Diferenciar un objeto de tipo Timeserie, produce un objeto timeserie
        differentiator = Differentitator()
        timeserie_diff = differentiator.transform(timeserie)
        
        #then
        self.assertTrue(isinstance(timeserie_diff, TimeSerie))
        
    def test_to_df(self):
        #given
        dummy_df = self.read_dummy_df()
        builder = DF2TimeSerie(pd.Timedelta('1d'))
        
        #when: Verifica que el objeto TImeserie esta soportado sobre un dataframe de pandas
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
        
        #when: Tranforma la función, asignandole otra función
        timeSerie_traforme = timeserie.apply(callback)
        df = timeSerie_traforme.to_df()
        
        #then
        self.assertTrue(isinstance(timeSerie_traforme,TimeSerie))
        self.assertEqual(dummy_df.shape[0],df.shape[0])

    def test_clone(self):
        #given
        dummy_df = self.read_dummy_df()
        builder = DF2TimeSerie(pd.Timedelta('1d'))
        timeserie = builder.build(dummy_df)

        #when: Crea otra copia de la data en memoria
        timeserie_clone = timeserie.clone()
        df = timeserie_clone.to_df()

        #then
        self.assertTrue(isinstance(timeserie_clone,TimeSerie))
        self.assertTrue(isinstance(df,pd.DataFrame))


unittest.main(argv=[''], verbosity=2, exit=False)

