# data
import pandas as pd
import numpy as np
from pandas_datareader import data as web
from datetime import datetime as dt
import functools

import config as cfg


def get_stockP_raw(l_of_stocks, start = dt(2020, 1, 1), end = dt.now()):
    # data preparation
    df = web.DataReader(l_of_stocks, 'yahoo', start, end)
    df = df.loc[:, df.columns.get_level_values(0).isin({'Close'})].round(4)
    df.columns =df.columns.droplevel()
    return df


def get_stockP_return(df_stockP):
    df_stock_return = df_stockP.pct_change().round(4)
    df_merged = df_stockP.merge(df_stock_return, 
                                left_index = True, 
                                right_index = True,
                                how = 'left',
                                suffixes=('_price', '_return'))
    return df_merged



def add_moving_features(df_raw):
    """ feature engineering """
    cols = df_raw.columns.tolist()
    
    # make statistical features
    l_new_features = []
    for col in cols:
        new_features = df_raw[col].rolling(5, min_periods=2).aggregate([np.min, np.max, np.mean]) #, np.std
        new_features = new_features.add_suffix('_{colname}'.format(colname = col))
        l_new_features.append(new_features)
    
    # add statistical features
    df_new_features = functools.reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how='left'), l_new_features)
    df_raw_moving_fts = pd.merge(df_raw, df_new_features, left_index=True, right_index=True, how='left')
    
    return df_raw_moving_fts
    
  
    
# get stock price
df_stockP_raw = get_stockP_raw(['0050.TW','VOO'],\
                               start = dt(cfg.stockp_start_yr, cfg.stockp_start_mn, cfg.stockp_start_d),\
                               end = dt(cfg.stockp_end_yr, cfg.stockp_end_mn, cfg.stockp_end_d))

# impute data
df_raw = df_stockP_raw.fillna(method='ffill')

# add returns
df_raw = get_stockP_return(df_raw)

# add moving features
df_raw = add_moving_features(df_raw)




# add columns
df_raw = df_raw.reset_index()
df_raw['Weekday'] = df_raw['Date'].dt.weekday.astype('category')
df_raw['Month'] = df_raw['Date'].dt.month.astype('category')
#df_raw['Weeknum'] = df_raw['Date'].dt.isocalendar().week.astype('category')
df_raw['Quarter'] = df_raw['Date'].dt.quarter.astype('category')
df_raw = df_raw.set_index('Date')

# one-hot encoding 
df_raw = pd.get_dummies(df_raw)


if __name__ == '__main__':
    print('Dataframe columns: ', df_raw.columns.tolist())



#df_raw.info()
#des = df.describe()




