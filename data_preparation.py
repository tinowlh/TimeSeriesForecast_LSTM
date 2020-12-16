# data
import pandas as pd
import numpy as np
from pandas_datareader import data as web
from datetime import datetime as dt
import functools



def get_stockP_raw(l_of_stocks, start = dt(2020, 1, 1), end = dt.now()):
    # data preparation
    df = web.DataReader(l_of_stocks, 'yahoo', start, end)
    df = df.loc[:, df.columns.get_level_values(0).isin({'Close'})].round(2)
    df.columns =df.columns.droplevel()
    return df


def get_stockP_return(df_stockP):
    df_stock_return = df_stockP.pct_change().round(4)
    return df_stock_return



def add_moving_features(df_raw):
    """ feature engineering """
    cols = df_raw.columns.tolist()
    
    # make statistical features
    l_new_features = []
    for col in cols:
        new_features = df_raw[col].rolling(7, min_periods=1).aggregate([np.min, np.max, np.mean]) #, np.std
        new_features = new_features.add_suffix('_{colname}'.format(colname = col))
        l_new_features.append(new_features)
    
    # add statistical features
    df_new_features = functools.reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how='left'), l_new_features)
    df_raw_moving_fts = pd.merge(df_raw, df_new_features, left_index=True, right_index=True, how='left')
    
    return df_raw_moving_fts
    
    

df_stockP_raw = get_stockP_raw(['0050.TW','VOO'], start = dt(2018, 12, 30), end = dt(2020, 12, 14))
df_raw = df_stockP_raw.fillna(method='ffill')
df_raw = add_moving_features(df_raw)


df_raw = df_raw.reset_index()
df_raw['Weekday'] = df_raw['Date'].dt.weekday
df_raw['Month'] = df_raw['Date'].dt.month
df_raw['Weeknum'] = df_raw['Date'].dt.week
df_raw['Quarter'] = df_raw['Date'].dt.quarter
df_raw = df_raw.set_index('Date')




#df_raw.info()
#des = df.describe()




