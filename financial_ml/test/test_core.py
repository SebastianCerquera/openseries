# -*- coding: utf-8 -*-
# +
import unittest
import pandas as pd

from financial_ml.core import TimeSerie, DF2TimeSerie, Differentitator, CSV2TimeSerie

# +
import os
import dotenv

dotenv.load_dotenv()
repo = os.environ.get('REPO')


# -

class DF2TimeSerieTest(unittest.TestCase):
   
    def test_default_columns(self):
        #give
        dummy_df = pd.read_parquet(repo + '/data/dummy.parquet')
        builder = DF2TimeSerie(pd.Timedelta('1d'))

        #when: Se construlle objeto Timeseria partiendo de un DataFrame
        timeserie = builder.build(dummy_df)

        #then
        self.assertTrue(isinstance(timeserie, TimeSerie))
    
    def test_default_custom_columsn(self):
        #give
        dummy_df = pd.read_parquet(repo + '/data/dummy_custom.parquet')
        builder = DF2TimeSerie(pd.Timedelta('1d'), date_column='Fecha', value_column='Valor Unidad')

        #when: Se construlle objeto Timeseria partiendo de un DataFrame
        timeserie = builder.build(dummy_df)

        #then
        self.assertTrue(isinstance(timeserie, TimeSerie)) 


class CSV2TimeSerieTest(unittest.TestCase):
    
    def test_read_series_defaults(self):
        #give
        builder = CSV2TimeSerie(repo + '/data/raw.csv', "test serie")

        #when: Se construlle objeto Timeseria partiendo de un DataFrame
        timeserie = builder.build(None)

        #then
        self.assertTrue(isinstance(timeserie, TimeSerie)) 


class TimeSerieTest(unittest.TestCase):
    
    @staticmethod
    def read_dummy_df():
        return pd.read_parquet(repo + '/data/dummy.parquet')
    
    @classmethod
    def setup(clazz):
        df = clazz.read_dummy_df()
        dummy_df = clazz.read_dummy_df()
        
        return dummy_df, DF2TimeSerie(pd.Timedelta('1d')).build(dummy_df)
        
    def test_transformation_differentiator(self):
        #given
        dummy_df, timeserie = self.setup()
        
        #when: Diferenciar un objeto de tipo Timeserie, produce un objeto timeserie
        differentiator = Differentitator()
        timeserie_diff = differentiator.transform(timeserie)
        
        #then
        self.assertTrue(isinstance(timeserie_diff, TimeSerie))
        
    def test_to_df(self):
        #given
        dummy_df, timeserie = self.setup()
        
        #when: Verifica que el objeto TImeserie esta soportado sobre un dataframe de pandas
        df = timeserie.to_df()
        
        #then
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertEqual(dummy_df.shape[0],df.shape[0])
        
    def test_apply(self):
        #given
        dummy_df, timeserie = self.setup()
        callback = lambda df:df
        
        #when: Tranforma la función, asignandole otra función
        timeSerie_traforme = timeserie.apply(callback)
        df = timeSerie_traforme.to_df()
        
        #then
        self.assertTrue(isinstance(timeSerie_traforme,TimeSerie))
        self.assertEqual(dummy_df.shape[0],df.shape[0])

    def test_clone(self):
        #given
        dummy_df, timeserie = self.setup()

        #when: Crea otra copia de la data en memoria
        timeserie_clone = timeserie.clone()
        df = timeserie_clone.to_df()

        #then
        self.assertTrue(isinstance(timeserie_clone,TimeSerie))
        self.assertTrue(isinstance(df,pd.DataFrame))


unittest.main(argv=[''], verbosity=2, exit=False)

