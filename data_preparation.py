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
df_raw['Weeknum'] = df_raw['Date'].dt.isocalendar().week
df_raw['Quarter'] = df_raw['Date'].dt.quarter
df_raw = df_raw.set_index('Date')





#df_raw.info()
#des = df.describe()




#### open data: ForeEX Rate
#def get_forex_rate(currency_cross, from_date, to_date, interval='Weekly'):
#
#    df = investpy.get_currency_cross_historical_data(currency_cross= currency_cross,
#                                        from_date= from_date,
#                                        to_date= to_date,
#                                        interval= interval)
#    df = df[['Close']]
#    df = df.rename(columns = {'Close':'{exrate}'.format(exrate=currency_cross)})
#    df.index = pd.to_datetime(df.index)
#    df.index = df.index.strftime("%Y") + df.index.strftime("%V")
#    df = df.groupby(df.index).mean()
#
#    return df
#
#


### EDA: Correlation Matrix ###
#df4CM = pd.merge(df, OECDE_data, left_index=True, right_index=True, how = 'left')
#corrmat = df4CM.corr() 
#covar = df4CM.cov()
#cg = sns.clustermap(corrmat, cmap ="YlGnBu", linewidths = 0.1); 
#plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation = 0) 
#plt.savefig('Z390 Sellin in Europe Correlation Matrix.png', dpi=200)
#cg 


### Tree-based model for selecting inputs ###
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split

#
#X = df.iloc[:,1:]
#y = df.iloc[:,0]
#
#scaler = StandardScaler()
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100)
#scaler.fit(X_train)
#X_train_transformed = scaler.transform(X_train)
#X_test_transformed = scaler.transform(X_test)
#
#
#rf = RandomForestRegressor(n_estimators = 400, max_depth = 8) #, max_features = 10
#rf.fit(X_train_transformed, y_train)
#
#
#y_pred = rf.predict(X_test)
#
#
## Evaluation
#rmse = sqrt(mean_squared_error(y_test, y_pred))
#
#
## Get numerical feature importances
#importances = rf.feature_importances_.tolist()
#
## List of tuples with variable and importance
#features = X.columns.tolist()
#
#df_FI = pd.DataFrame(list(zip(features, importances)), 
#               columns =['features', 'importances']) 
#
#
#df_FI = df_FI.sort_values(by='importances', ascending=False)





