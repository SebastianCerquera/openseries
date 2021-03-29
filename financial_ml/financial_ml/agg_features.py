from .core import *


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

__BUILDER__ = ['WeeklyExtractor', 'MonthlyExtractor', 'YearlyExtractor']


