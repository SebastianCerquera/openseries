# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

from datetime import datetime
from financial_ml.io import *


# +
## Builders

class ValoresBancolombiaCSV(CSV2TimeSerie):    

    months_map = {
        'enero': '01',
        'febrero': '02',
        'marzo': '03',
        'abril': '04',
        'mayo': '05',
        'junio': '06',
        'julio': '07',
        'agosto': '08',
        'septiembre': '09',
        'octubre': '10',
        'noviembre': '11',
        'diciembre': '12'
    }

    @classmethod
    def valores_bancolombia_date_pattern(cls, date):
        parts = date.split('/')
        if parts[1] not in cls.months_map.keys():
            print(date)
            raise Exception()
        return datetime.strptime(
            parts[0] + "/" + cls.months_map[parts[1]] + "/" + parts[2], "%d/%m/%Y")
    
    def __init__(self, filename, serie_name):
        super().__init__(
            filename,
            serie_name,
            serie_column=None,
            fund_column=None,
            date_column="FECHA",
            value_column="VALOR UNIDAD",
            date_pattern=self.valores_bancolombia_date_pattern
        )

__BUILDER__ = ['ValoresBancolombiaCSV']


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



__all__ =  __BUILDER__ + __VISUALIZATIONS__
