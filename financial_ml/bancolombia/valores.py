# -*- coding: utf-8 -*-
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
            parts[0] + "/" + cls.months_map[parts[1]] + "/" + parts[2], "%Y/%m/%d")
    
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
# -

__all__ =  __BUILDER__
