from data_preparation import df_raw

import pandas as pd 
import pickle
from keras.models import load_model
import numpy as np



# convert time series into inputs
def series_to_input_production(data, n_in=1, dropnan=True): 

	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = [], []

	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i-1))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names

    # drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)

	return agg


# create a differenced series (make data stationary: remove trend)
def difference(dataset, interval=1):

	diff = []
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)

	return np.vstack(diff)


# transform series into inputs for prediction (with differencing)
def prepare_data_production(df, scaler_load, n_lag):

	# extract raw values
    raw_values = df.values.astype('float32')

	# transform data to be stationary
    diff_values = difference(raw_values, 1)

	# rescale values to -1, 1
    scaled_values = scaler_load.transform(diff_values) 
    scaled_values = scaled_values[-n_lag:]

	# transform into supervised learning problem X, y
    inputs = series_to_input_production(scaled_values, n_lag)
    inputs_values = inputs.values

    return inputs_values


# make one forecast with an LSTM,
def forecast_lstm(model, test_X, n_batch):

	# reshape input pattern to [samples, timesteps, features]
    test_X = test_X.reshape((1, n_lag, n_features))

	# make forecast
    forecast = model.predict(test_X, batch_size=n_batch)

    # convert to array
    return [x for x in forecast[0, :]]



# make forecasts
def make_forecasts(model, n_batch, test, n_lag, n_seq):

    forecasts = []
    n_features = df.shape[1]
    n_obs = n_lag * n_features
    for i in range(len(test)):        
        test_X = test[i, :n_obs]

		# make forecast
        forecast = forecast_lstm(model, test_X, n_batch)

		# store the forecast
        forecasts.append(forecast)

    return forecasts



# invert differenced forecast
def inverse_difference(last_ob, forecast):

 	# invert first forecast
 	inverted = []
 	inverted.append(forecast[0] + last_ob)

 	# propagate difference forecast using inverted first value
 	for i in range(1, len(forecast)):
 		inverted.append(forecast[i] + inverted[i-1])

 	return inverted



# inverse data transform on forecasts
def inverse_transform(df, forecasts, scaler, n_test):

   inverted = []
   for i in range(len(forecasts)):

		# create array from forecast
        forecast = np.array(forecasts[i])

        # forecast = forecast.reshape(1, len(forecast))
        forecast = forecast.reshape(len(forecast),1)

        # temp for inverse_transform
        temp = np.arange(n_seq * (n_features-1)).reshape(n_seq,(n_features-1))
        forecast_t = np.c_[forecast,temp]

		# invert scaling 
        inv_scale = scaler.inverse_transform(forecast_t)
        inv_scale = inv_scale[:, 0]

		# invert differencing
        index = len(df) - n_test + i - 1
        last_ob = df.values[index, 0]
        inv_diff = inverse_difference(last_ob, inv_scale)

		# store
#        inverted.append(inv_scale) # without differencing
        inverted.append(inv_diff)
        

   return inverted



## forecast (production)
def forecast_prod(df, n_lag, n_seq, n_batch, model_load, scaler_load):

    # prepare data
    inputs_values = prepare_data_production(df, scaler_load, n_lag)
    
    # make forecasts   
    forecasts_prod = make_forecasts(model_load, n_batch, inputs_values, n_lag, n_seq)
        
    # inverse transform forecasts
    forecasts_inver_prod = inverse_transform(df, forecasts_prod, scaler = scaler_load, n_test = 1 + (n_seq-1))
    
        
    return forecasts_inver_prod



### START ###
    
### Initialize saved files ###
scaler_load = pickle.load(open('./model/scaler_v1.0.pickle', 'rb'))
model_load = load_model('./model/LSTM_v1.0.h5')   

df = df_raw.copy()

# model config
n_lag = 10 # input
n_seq = 1 # output
n_test = 10
n_epochs = 20
n_batch = 32
n_neurons = 8
n_features = df.shape[1]
n_obs = n_lag * df.shape[1]

if __name__ == '__main__':
    
    forecasts = forecast_prod(df, n_lag, n_seq, n_batch, model_load, scaler_load)
    print(f'0050.TW {df.index[-1].date()} price: {df.iloc[-1,0]} \nNext day price forcasting: {round(forecasts[0][0],2)}')
