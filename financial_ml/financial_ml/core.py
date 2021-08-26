# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# +
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
# -



from abc import ABC, abstractmethod


# +
## Declares entities
# -

class TimeSerie(ABC):
    
    def __init__(self, history, freq):
        assert isinstance(freq, pd.Timedelta), "The frequency should be a pandas Timedelta"
        assert isinstance(history, pd.DataFrame), "The frequency should be a pandas Timedelta"

        self.history = history.sort_values(by='ds').reset_index(drop=True)
        self.freq = freq
        
    def clone(self, history=None):
        if history is None:
            history = self.history
        return self.__class__(history, self.freq)
    
    def apply(self, callback):
        df = callback(self.history)
        return self.clone(history=df)
    
    def subset(self, prediction_dates):
        prediction_min = prediction_dates['ds'].min()
        prediction_max = prediction_dates['ds'].max()
        history = self.history.loc[
            (self.history['ds'] >= prediction_min) & (self.history['ds'] <= prediction_max)
        ]
        return self.clone(history=history)
    
    def to_df(self):
        return self.history.copy()



# +
"""
I still don't know what possible methods could provide a random varible but i do know that a ml algorithm 
is estimating a parameter, i am going to evaluate my estimator against some hold out data. It could be handy to
have some method to easily compare them.

Both my label and the model result should share the same instance type.
"""

class RandomVariable(ABC):
    
    def __init__(self, index, values):
        self.index = index
        self.values = values
        
    def to_df(self):
        return pd.DataFrame({
            'value': self.values
        }, index=self.index)
    
class ContinousVariable(RandomVariable):
    pass
    
class DiscreteVariable(RandomVariable):
    pass


# +
class Features(ABC):
    
    @abstractmethod
    def features_names(self):
        pass
    
    @abstractmethod
    def load(self, name, source):
        pass
    
    @abstractmethod
    def to_df(self):
        pass
    
    @abstractmethod
    def write(self, drop):
        pass
    
class DataFrameFeatures(Features):

    def __init__(self, features):
        assert isinstance(features, pd.DataFrame)
        self.features = features

    def features_names(self):
        return self.features.columns
    
    def load(self, name, source):
        raise Exception()
    
    def to_df(self):
        return self.features.copy()
    
    def write(self, drop):
        ## TODO I just add the stub implemenation to make it run
        raise Exception()
    
class EDAFeatures(Features):
    
    @staticmethod    
    def transformations_to_extractors(extractors):
        extractors_dict = {}
        for extractor in extractors:
            assert isinstance(extractor, FeatureExtractor)
            extractors_dict[extractor.name] = extractor
        return extractors_dict
    
    def __init__(self, extractors):
        self.features = {}
        self.extractors = self.transformations_to_extractors(extractors)

    def features_names(self):
        return list(self.extractors.keys())
    
    def load(self, name, source):
        self.features[name] = self.extractors[name].transform(source) 
    
    @staticmethod      
    def get_features_df(name, features):
        features_df = features[name].to_df()
        features_df.loc[:, name] = features_df['value']
        features_df.loc[:, 'ds'] = features_df.index
        return features_df.drop(['value'], axis=1).reset_index(drop=True)
    
    def to_df(self):
        keys = self.features_names()
        final = self.get_features_df(keys[0], self.features)
        for i in range(1, len(keys)):
            final = final.merge(self.get_features_df(keys[i], self.features), on='ds')
        final= final.sort_values(by='ds')
        return final.drop(['ds'], axis=1)


# -
class Label(ABC):    
    
    def __init__(self, variable):
        assert isinstance(variable, RandomVariable)
        self.variable = variable
    
    def to_df(self):
        df = self.variable.to_df()
        df.loc[:, 'ds'] = df.index
        df = df.sort_values(by='ds')
        return df.drop(['ds'], axis=1).reset_index(drop=True)
    
    @abstractmethod
    def write(self, drop):
        pass


__ENTITIES__ = [
    'TimeSerie', 'Features', 'DataFrameFeatures', 'EDAFeatures', 'RandomVariable', 
    'ContinousVariable', 'DiscreteVariable', 'Label'
]


class Feature(ABC):
    
    def __init__(self, name, variable):
        assert isinstance(variable, RandomVariable)
        
        self.name = name
        self.variable = variable
        
    def to_df(self):
        df = self.variable.to_df()
        df.loc[:, self.name] = df['value']
        return df.drop(['value'], axis=1)




class Strategy(ABC):
    pass


class BackTest(ABC):
    pass





# +
# Declare builders

class BaseBuilder(ABC):
    
    @abstractmethod
    def build(self, data):
        pass
# -



class TimeSerieBuilder(BaseBuilder):
    pass


class DF2TimeSerie(TimeSerieBuilder):
    
    def __init__(self, freq, date_column='ds', value_column='y'):
        self.freq = freq
        
        self.date_column = date_column
        self.value_column = value_column
        
    def safe_series(self, df):
        df = df.copy().sort_values(by=self.date_column)
        
        intervals = df.iloc[1:].reset_index(drop=True) - \
                df.iloc[:-1].reset_index(drop=True)
        intervals = intervals[self.date_column].value_counts()
        
        assert intervals.shape[0] == 1, "there are missing points"
        assert intervals.index[0] == self.freq, "the interval must match the frequency"
        
        df = df.copy()
        df.loc[:, "ds"] = df[self.date_column]
        df.loc[:, "y"] = df[self.value_column]
        
        return df
        
    def build(self, data):
        assert isinstance(data, pd.DataFrame), "The data parameter should be a pandas DataFrame"
        assert self.date_column in data.columns, f"The dataframe is missing the column: {self.date_column}"
        assert self.value_column in data.columns, f"The dataframe is missing the column: {self.value_column}"
        
        df = self.safe_series(data)
        
        return TimeSerie(df[["ds", "y"]], self.freq)


class Array2TimeSerie(DF2TimeSerie):
    
    @staticmethod
    def generate_series_dates(series, start_date, freq):
        assert isinstance(series, numpy.ndarray), "The serie paramater should a numpy array"
        assert series.shape[0] > 0, "The serie is empty"
        
        dates_seq = pd.date_range(
            start=start_date,
            periods=series.shape[0],
            freq=self.freq
        )
        
        return pd.DataFrame({
            'ds': dates_seq,
            'y': series
        })
    
    def __init__(self, start_date, freq):
        super().__init__(freq)
        
        self.start_date = start_date
        
    def build(self, data):
        df = generate_series_dates(data, self.start_date, self.freq)
        
        return super().build(df)


__TIMESERIES_BUILDERS__ = ['TimeSerieBuilder', 'DF2TimeSerie', 'Array2TimeSerie' ]


class TimeSerie2Variable(BaseBuilder):
    
    def __init__(self, continuous=False):        
        self.builder = DiscreteVariable
        if continuous:
            self.builder = ContinousVariable
        
    def build(self, data):
        assert isinstance(data, TimeSerie)
        df = data.to_df()
        return self.builder(df['ds'].values, df['y'].values)


class DF2Features(BaseBuilder):
    
    def build(self, df):
        assert isinstance(df, pd.DataFrame)
        return DataFrameFeatures(df)


__FEATURE_BUILDERS__ = ['TimeSerie2Variable', 'DF2Features']


# +
class TimeSerie2Feature(TimeSerie2Variable):
    
    def __init__(self, name, continuous=False):
        super().__init__(continuous)
        self.name = name
    
    def build(self, data):
        variable = super().build(data)
        return Feature(self.name, variable)

class TimeSerie2Label(TimeSerie2Variable):
    
    def build(self, data):
        variable = super().build(data)
        return Feature(self.name, variable)
# -



# +
class FeaturesBuilder(ABC):
    
    def __init__(self, features):
        assert isinstance(features, Features), "The feature extractor needs "
        
        self.features = features

    def extract(self, data):
        for name in self.features.features_names():
            self.features.load(name, data)

__FEATURES_BUILDERS__ = ['FeaturesBuilder']
# -

__BUILDERS__ = __TIMESERIES_BUILDERS__ + __FEATURE_BUILDERS__ + __FEATURES_BUILDERS__








# +
# Declare validators
# -

class BackTester(ABC):
    
    @abstractmethod
    def run_strategy(self, strategy):
        pass





# +
# Declare preprocessors

# +
class Transformation(ABC):
    
    @abstractmethod
    def transform(self, data):
        pass
    
    
class Differentitator(Transformation):
    
    @staticmethod
    def diff_df(df):
        df = df.copy().sort_values(by='ds')
        
        diff = df['y'].iloc[1:].reset_index(drop=True) - \
                df['y'].iloc[:-1].reset_index(drop=True)
        
        final = df.iloc[1:].reset_index(drop=True)
        final.loc[:, 'y'] = diff/df['y'].iloc[:-1].reset_index(drop=True)
        
        filled = final.loc[final['y'].apply(lambda e: e != np.inf and e != -np.inf), 'y'].mean()

        final.loc[final['y'].isna(), 'y'] = 0
        final.loc[final['y'].apply(lambda e: e == np.inf), 'y'] = filled
        final.loc[final['y'].apply(lambda e: e == -np.inf), 'y'] = filled
        
        return final
        
    
    def transform(self, data):
        assert isinstance(data, TimeSerie), "The differentiator only works over time series"
        return data.apply(self.diff_df)
    
class CustomArange(Transformation):
    
    def __init__(self, N):
        self.N = N
    
    def custom_arange(self, df):
        a = df['ds'].shape[0]/self.N
        b = np.array(list(map(lambda e: int(e*a) - 1, range(self.N+1))))
        if max(b) > max(df.index):
            return df['ds'].loc[df.index[b[:-1]]].values
        return df['ds'].loc[df.index[b]].values
        
    def transform(self, data):
        assert isinstance(data, TimeSerie), "The differentiator only works over time series"
        return self.custom_arange(data.to_df())
    
class ExpandDate(Transformation):
    
    @staticmethod
    def expand(timeserie):
        df = timeserie.to_df()        
        df.loc[:, "year"] = df["ds"].apply(lambda e: e.year)
        df.loc[:, "month"] = df["ds"].apply(lambda e: e.month)
        df.loc[:, "weekday"] = df["ds"].apply(lambda e: e.weekday())
        return df

    def transform(self, data):
        assert isinstance(data, TimeSerie), "The differentiator only works over time series"    
        return self.expand(data)
        
class TimeSerie2Hist(Transformation):
    
    def __init__(self, bins):
        super().__init__()
        
        self.bins = bins
    
    def build_hist_table(self, serie):
        x_min = min(serie)
        x_max = max(serie)
        step = (x_max - x_min)/float(self.bins)
        bins = np.arange(x_min, x_max, step)
        serie_count = len(serie)
        hist = np.histogram(serie, bins)
        hist_proportion = list(map(lambda e: e/serie_count, hist[0]))
        final = pd.DataFrame({'bin': hist[1][1:], 
                              'count':hist[0], 'proportion':hist_proportion})
        final.loc[:, "mean"] = np.mean(serie)
        final.loc[:, "std"] = np.std(serie)
        final.loc[:, "median"] = np.median(serie)
        return final

    def transform(self, data):
        assert isinstance(data, TimeSerie), "The differentiator only works over time series"    
        df = data.to_df()
        return self.build_hist_table(df['y'])
    
class TimeSerieRange(Transformation):

    def transform(self, data):
        assert isinstance(data, TimeSerie), "The differentiator only works over time series"    
        df = data.to_df()
        return {
            'min': df['y'].min(),
            'max': df['y'].max()
        }


# -
__BASE_PREPROCESSORS__ = ['Transformation', 'Differentitator', 'CustomArange', 'ExpandDate', 'TimeSerie2Hist', 'TimeSerieRange']


# +
class CumulativeStatistics(Transformation):

    def __init__(self, name, periods, estimator):
        self.periods = periods
        self.estimator = estimator
        self.name = name
    
    def transform(self, data):
        assert isinstance(data, TimeSerie), "The differentiator only works over time series"
        
        df = data.to_df()
        if df.shape[0] < self.periods:
            #raise AssertionError("There are not enough points to aggregated by this period: {}".format(self.periods))
            raise Exception()

        df.loc[:, 'start_date'] = df['ds']
        
        exploded = []
        for i in df.iloc[self.periods:].index:
            row = []
            for j in range(self.periods):
                row.append([
                    df.iloc[i]['ds'],
                    df.iloc[i - j]['y']
                ])
            tx = pd.DataFrame(row)
            tx.columns = ['start_date', 'y']
            exploded.append(tx)
        exploded = pd.concat(exploded).reset_index()
        
        estimated = exploded[['start_date', 'y']].groupby('start_date').agg(self.estimator)
        estimated.loc[:, 'ds'] = estimated.index
        
        estimated = estimated.loc[:, ['ds', 'y']].reset_index(drop=True)
        estimated.columns = ['ds', self.name]
        return estimated
    
class FeatureExtractor(CumulativeStatistics):
    
    def __init__(self, name, period, callback):
        super().__init__(name, period, callback)


# -

__FEATURES_PREPROCESSORS__ = ['CumulativeStatistics', 'FeatureExtractor']

__PREPROCESSORS__ = __BASE_PREPROCESSORS__ + __FEATURES_PREPROCESSORS__




# +
# Declare models

class Classifier(ABC):
    pass

class Regressor(ABC):
    
    @abstractmethod
    def predict(self, features):
        pass
    
class BaseRegressor(Regressor):
    
    def predict(self, features):
        assert isinstance(features, Features), "The regressor can only use features to predict"


# -

__MODELS__ = ['Classifier', 'Regressor']



# +
# Visualizations
    
class PyplotTimeSerie(object):
    
    def __init__(self, figsize=(15, 20)):
        self.fig = plt.figure(constrained_layout=True, figsize=figsize)
        self.grid = self.fig.add_gridspec(9, 4)
    
    def plot_series(self, timeserie, grid, title, label=None, x="ds", y="y"):
        ax = self.fig.add_subplot(grid)
        ax.set_title(title)
        
        
        xticks = CustomArange(10).transform(timeserie)
        df = timeserie.to_df()
        #ax.grid(color='r', linestyle='-', linewidth=0.1)
        
        if label is None:
            ax.plot(df[x], df[y])
        else:
            ax.plot(df[x], df[y], label=label)
            ax.legend()
        
        ax.set_xticks(xticks)
       
    def plot_series_interval(self, timeserie):
        self.plot_series(timeserie, self.grid[0, :], "Unit value")
        
        differentiator = Differentitator()
        timeserie_diff = differentiator.transform(timeserie)
        
        self.plot_series(timeserie_diff, self.grid[1, :], "Relative variations")
        
    def plot_boxplot(self, df, grid, title, x="x", y="y", outliers=False):
        ax = self.fig.add_subplot(grid)
        ax.set_title(title)
        sns.boxplot(ax=ax, data=df, y=y, x=x, showfliers=outliers)
        
    def plot_boxplot_with_diff(self, timeserie, label, grid_index, message, inline=False):
        differentiator = Differentitator()
        timeserie_diff = differentiator.transform(timeserie)
        
        unit_df = ExpandDate().transform(timeserie)
        if inline:
            self.plot_boxplot(unit_df, self.grid[grid_index, :2], message, x=label, y="y")
        else:
            self.plot_boxplot(unit_df, self.grid[grid_index, :], message, x=label, y="y")
        
        diff_df = ExpandDate().transform(timeserie_diff)
        if inline:
            self.plot_boxplot(diff_df, self.grid[grid_index, 2:], message, x=label, y="y")
        else:
            self.plot_boxplot(diff_df, self.grid[grid_index + 1, :], message, x=label, y="y")

    @classmethod
    def plot_hist_axis(cls, ax, df, bounds, bins, max_value=None):
        width=(bounds['max'] - bounds['min'])/float(bins)
        ax.bar(df["bin"], df["count"], width=width)
        if max_value != None:
            ax.set_xlim(0, max_value)
        ax.grid(color='r', linestyle='-', linewidth=0.5)
    
    ## TODO hace referencia a una función externa
    def plot_hist(self, timeserie, grid, title, N=30):
        ax = self.fig.add_subplot(grid)
        ax.set_title(title)
        
        df = TimeSerie2Hist(N).transform(timeserie)
        bounds = TimeSerieRange().transform(timeserie)

        ax.grid(color='r', linestyle='-', linewidth=0.5)
        self.plot_hist_axis(ax, df, bounds, 30, max_value=None)
        
    def plot_hist_variations(self, timeserie, y="y"):
        differentiator = Differentitator()
        timeserie_diff = differentiator.transform(timeserie)

        self.plot_hist(timeserie_diff, self.grid[7,:], "Distribución de la variación")

    def plot_autocorrelation(self, y, grid, lag, title, N=30):
        ax = self.fig.add_subplot(grid)
        ax.set_title(title)
        
        ax.grid(color='r', linestyle='-', linewidth=0.5)
        a = sm.graphics.tsa.plot_pacf(y, lags=lag, ax=ax)

    def plot_autocorrelation_variations(self, timeserie, y="y"):
        differentiator = Differentitator()
        timeserie_diff = differentiator.transform(timeserie)
        diff = timeserie_diff.to_df()
        
        if diff.shape[0] > 7:
            a = self.plot_autocorrelation(diff[y], self.grid[8,0], 7, "Autocorrelation 7 días")
        
        if diff.shape[0] > 15:
            a = self.plot_autocorrelation(diff[y], self.grid[8,1], 15, "Autocorrelation 15 días")
        
        if diff.shape[0] > 30:
            a = self.plot_autocorrelation(diff[y], self.grid[8,2], 30, "Autocorrelation 30 días")
    
    def plot(self, timeserie):
        self.plot_series_interval(timeserie)
        
        self.plot_boxplot_with_diff(timeserie, "year", 2, "Distribución anual")
        self.plot_boxplot_with_diff(timeserie, "month", 4, "Distribución anual", inline=True)
        self.plot_boxplot_with_diff(timeserie, "weekday", 5, "Distribución anual", inline=True)
        
        self.plot_hist(timeserie, self.grid[6,:], "Distribución del valor de la unidad")

        self.plot_hist_variations(timeserie)
        self.plot_autocorrelation_variations(timeserie)
# -

__VISUALIZATIONS__ = ['PyplotTimeSerie']



__all__ = __VISUALIZATIONS__ + __ENTITIES__ + __BUILDERS__ + __PREPROCESSORS__ + __MODELS__
