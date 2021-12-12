# +
import os
import csv
import codecs

from collections import Callable

import pandas as pd
from datetime import datetime

from .core import *


# -

class CSV2TimeSerie(TimeSerieBuilder):    

    @staticmethod
    def read_csv_with_encoding(filename, delimiter="|", encoding="utf-8"):
        with codecs.open(filename, encoding=encoding) as fp:
            reader = csv.reader(fp, delimiter=delimiter)
            csvFile = list(reader)
            return pd.DataFrame(csvFile[1:], columns=csvFile[0])
        
    def format_date_column(self, series, pattern):
        if type(pattern) == str:
            return series.apply(lambda e: datetime.strptime(e, pattern))
        elif isinstance(pattern, Callable):
            return series.apply(lambda e: pattern(e))
        else:
            raise Exception()
        
    def prepare_dataframe(self, df):
        df = df.copy()
        df.loc[:, self.date_column] = self.format_date_column(df[self.date_column], self.date_pattern)
        df.loc[:, self.value_column] = df[self.value_column].apply(lambda e: e.replace('$', ''))
        df.loc[:, self.value_column] = df[self.value_column].apply(lambda e: e.replace(',', ''))
        df.loc[:, self.value_column] = df[self.value_column].apply(float)
        return df
        
    def investors_df(self, df):
        if self.fund_name is not None and self.fund_column in df.columns:
            df = df.loc[df[self.fund_column] == self.fund_name]
            
        if self.serie_column is not None and self.serie_column in df.columns:
            df = df.loc[df[self.serie_column] == self.serie_name]

        df = df.loc[:, [self.date_column, self.value_column]]
        return self.prepare_dataframe(df)
    
    def __init__(self, filename, serie_name, fund_name=None,
                 serie_column="Nombre_Fondo", fund_column="Nombre_Entidad",
                 date_column="Fecha corte", date_pattern="%d/%m/%Y",
                 period="1d", value_column="Valor Unidad", delimiter="|"):
        self.filename = filename
        self.serie_name = serie_name
        self.fund_name = fund_name

        self.serie_column = serie_column
        self.fund_column = fund_column
        self.date_column = date_column
        self.date_pattern = date_pattern
        self.period = period
        self.value_column = value_column
        self.delimiter = delimiter

    def build(self, data):
        df = self.read_csv_with_encoding(self.filename, delimiter=self.delimiter)
        df = self.investors_df(df)
             
        return DF2TimeSerie(
            pd.Timedelta(self.period),
            date_column = self.date_column,
            value_column = self.value_column
        ).build(df)


__all__ = ['CSV2TimeSerie']
