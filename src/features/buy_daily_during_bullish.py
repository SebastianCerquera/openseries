# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

# +
import os
import csv
import codecs

import numpy as np
import pandas as pd
from datetime import datetime 

import seaborn as sns
import matplotlib.pyplot as plt

from financial_ml.core import *


# +
## Builders

class CSV2TimeSerie(TimeSerieBuilder):    

    @staticmethod
    def read_csv_with_encoding(filename, delimiter="|", encoding="utf-8"):
        with codecs.open(filename, encoding=encoding) as fp:
            reader = csv.reader(fp, delimiter=delimiter)
            csvFile = list(reader)
            return pd.DataFrame(csvFile[1:], columns=csvFile[0])

    @staticmethod
    def investors_df(df, fund, ds, value, name):
        df = df.loc[:, [ds, value, name]]
        df.loc[:, "ds"] = df[ds].apply(lambda e: datetime.strptime(e, "%d/%m/%Y"))
        df.loc[:, "y"] = df[value].apply(lambda e: e.replace('$', ''))
        df.loc[:, "y"] = df["y"].apply(lambda e: e.replace(',', ''))
        df.loc[:, "y"] = df["y"].apply(float)
        df = df.loc[df[name] == fund]
        return df.drop([ds, value, name], axis=1)
    
    def __init__(self, filename, serie_name):
        self.filename = filename
        self.serie_name = serie_name
        self.date_name = "Fecha corte"
        self.fund_name = "Nombre Negocio"
        
    def build(self, data):
        df = self.read_csv_with_encoding(self.filename)
        df = self.investors_df(df, data, self.date_name, 
                               self.serie_name, self.fund_name)
                               
        return DF2TimeSerie(pd.Timedelta('1d')).build(df)

class ContinousExtractor(FeatureExtractor):
     
    def transform(self, data):
        df = super().transform(data)
        
        return ContinousVariable(df['ds'].values, df[self.name].values)

class WeeklyExtractor(ContinousExtractor):
    
    def __init__(self, name, callback):
        super().__init__(name, 7, callback)
        
class MonthlyExtractor(ContinousExtractor):
    
    def __init__(self, name, callback):
        super().__init__(name, 30, callback)
        
class YearlyExtractor(ContinousExtractor):
    
    def __init__(self, name, callback):
        super().__init__(name, 360, callback)
    
__BUILDER__ = ['CSV2TimeSerie', 'ContinousExtractor', 'WeeklyExtractor', 'MonthlyExtractor',  'YearlyExtractor']

# +
## Entities

import os
import dotenv

class SerializerHelper(ABC):
    
    stressed_vocals = {
        'á': 'a',
        'é': 'e',
        'í': 'i',
        'ó': 'o',
        'ú': 'u'
    }

    special_characters = {
        ' ': '_',
        '.': ''
    }

    @classmethod
    def safe_path_string(cls, text):
        for k, v in cls.stressed_vocals.items():
            text = text.replace(k, v)
        for k, v in cls.special_characters.items():
            text = text.replace(k, v)
        return text
    
    @staticmethod
    def get_repo_path():
        dotenv.load_dotenv()
        return os.environ.get('REPO')
    
    @classmethod
    def safe_to_parquet(cls, df, fund, date_suffix, source, filename):
        fund = cls.safe_path_string(fund)
        
        repo = cls.get_repo_path()
        execution_home = f'{repo}/data/{date_suffix}/{fund}/{source}/'
        
        if not os.path.exists(execution_home):
            os.makedirs(execution_home)
        filename = execution_home + f'{filename}.parquet'
        print(filename)
        df.to_parquet(filename, index=False)
        
    def __init__(self, source, fund_name, serie_name, date_suffix, output_suffix):
        self.source = source
        self.fund_name = fund_name
        self.serie_name = serie_name
        self.date_suffix = date_suffix
        self.output_filename = f"{output_suffix}__" + self.safe_path_string(serie_name) 

    def write_df(self, df, drop=[]):
        assert isinstance(df, pd.DataFrame)
        assert isinstance(drop, list), "The drop argument should be the list of features to ommit"
        
        df = df.drop(drop, axis=1)
        
        self.safe_to_parquet(
            df, 
            self.fund_name, 
            self.date_suffix, 
            self.source, 
            self.output_filename
        )

class EDAFeaturesImpl(EDAFeatures):
    
    def __init__(self, extractors, fund_name, serie_name, date_suffix, output_suffix):
        super().__init__(extractors)
        
        self.serializer = SerializerHelper(
            "features",
            fund_name,
            serie_name,
            date_suffix,
            output_suffix
        )
        
    def write(self, drop):
        df = self.to_df()
        self.serializer.write_df(df, drop=drop)
        
class LabelImp(Label):
    
    def __init__(self, variable, fund_name, serie_name, date_suffix, output_suffix):
        super().__init__(variable)
        
        self.serializer = SerializerHelper(
            "labels",
            fund_name,
            serie_name,
            date_suffix,
            output_suffix
        )
        
    def write(self, drop):
        df = self.to_df()
        self.serializer.write_df(df, drop=drop)
        
__ENTITIES__ = ['EDAFeaturesImpl', 'LabelImp']


# +
class PyplotVariable(ABC):
    
    def __init__(self, figsize, grid_y, grid_x):
        self.fig = plt.figure(constrained_layout=True, figsize=figsize)
        self.grid = self.fig.add_gridspec(grid_y, grid_x)
        
    @abstractmethod
    def plot(self, variable):
        pass
    
class PyplotDiscrete(PyplotVariable):
    
    def __init__(self):
        super().__init__((15, 10), 2, 3)
    
    def plot(self, variable):
        pass
    
class PyplotContinuous(PyplotVariable):
    
    def __init__(self, title):
        super().__init__((15, 8), 2, 3)
        
        self.fig.suptitle(title, fontsize=20)
    
    def plot_distribution(self, serie, row):
        # Histogram graph
        ax = self.fig.add_subplot(self.grid[row, 0])
        ax.set_title("Histogram")
        sns.distplot(serie, ax=ax)

        # Boxplot graph
        ax = self.fig.add_subplot(self.grid[row, 1])
        ax.set_title("Boxplot")
        sns.boxplot(serie, ax=ax)
        
        # Violin plot
        ax = self.fig.add_subplot(self.grid[row, 2])
        ax.set_title("Violinplot")
        sns.violinplot(serie, ax=ax)

    @staticmethod
    def filter_atipical(df):
        q1 = np.percentile(df['value'], 25)
        q2 = np.percentile(df['value'], 50)
        q3 = np.percentile(df['value'], 75)

        iqd = q3 - q1
        too_low = q1 - 1.5*iqd
        too_high = q3 + 1.5*iqd

        val = df['value']
        return df.loc[(val >= too_low) &  (val <= too_high)].reset_index(drop=True)
        
    
    def plot(self, variable):
        df = variable.to_df()
        self.plot_distribution(df['value'], 0)
        
        tipical = self.filter_atipical(df)
        self.plot_distribution(tipical['value'], 1)
        
class PyplotFeatures(ABC):
        
    def plot(self, features):
        assert isinstance(features, Features)
        
        for name in features.features_names():
            PyplotContinuous(name).plot(features.features[name])

__VISUALIZATIONS__ = ['PyplotContinuous', 'PyplotFeatures', 'PyplotDiscrete']
# -

__all__ =  __BUILDER__ + __ENTITIES__ + __VISUALIZATIONS__
