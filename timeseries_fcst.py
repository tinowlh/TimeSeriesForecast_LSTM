# set seed to get reproducible results
from numpy.random import seed
seed(1000)
import tensorflow as tf
tf.random.set_seed(1000)


# data
from data_preparation import df_raw
import pandas as pd 
import pickle


# model building
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from keras.models import Sequential
from keras.layers import LSTM, Dense #, GRU
from keras.models import load_model
from keras.optimizers import Adam #, RMSprop
#from keras.callbacks import EarlyStopping


# model evaluation
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
import seaborn as sns



# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = [], []

	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

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

# transform series into train and test sets for supervised learning (with differencing)
def prepare_data(df, n_test, n_lag, n_seq):
    
	# extract raw values
    raw_values = df.values.astype('float32')
	# transform data to be stationary
    diff_values = difference(raw_values, 1)
	# rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(diff_values)
	# transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values
	# split into train and test sets
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]

    return scaler, train, test
    

# transform series into train and test sets for supervised learning (without differencing)
#def prepare_data(df, n_test, n_lag, n_seq):
#    
#	# extract raw values
#    raw_values = df.values.astype('float32')
#	# rescale values to -1, 1
#    scaler = MinMaxScaler(feature_range=(0, 1))
#    scaled_values = scaler.fit_transform(raw_values)
#	# transform into supervised learning problem X, y
#    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
#    supervised_values = supervised.values
#	# split into train and test sets
#    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
#
#    return scaler, train, test


def split_into_input_output(train, test, n_seq):

	# split into input and output
    n_features = df.shape[1]   
    n_obs = n_lag * n_features

    # _y 取T+1 ~ T+6 
    y_col = [(i + 1) * -n_features for i in reversed(range(n_seq))]
    train_X, train_y = train[:, :n_obs], train[:, y_col]
    test_X, test_y = test[:, :n_obs], test[:, y_col]

    # reshape training into [samples, timesteps, features]    
    train_X = train_X.reshape((train_X.shape[0], n_lag, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_lag, n_features))

    return train_X, train_y, test_X, test_y

    
def fit_lstm(train_X, train_y, test_X, test_y, n_lag, n_seq, n_batch, nb_epoch, n_neurons):

	# design network
    model = Sequential()
    model.add(LSTM(n_neurons, 
                   dropout = 0.5,
#                   recurrent_dropout = 0.3,
#                   kernel_regularizer=l1(0.0000001), #l1_l2(0.0000001, 0.0000001), #l2(0.0000001),
#                   recurrent_regularizer= l1(0.0000001), #l1_l2(0.0000001, 0.0000001), #l2(0.0000001),
                   input_shape=(train_X.shape[1], train_X.shape[2]), 
                   return_sequences=True
                   ))
    model.add(LSTM(n_neurons, 
                   dropout = 0.5,
#                   recurrent_dropout = 0.3,
#                   kernel_regularizer=l1(0.0000001), #l1_l2(0.0000001, 0.0000001), #l2(0.0000001),
#                   recurrent_regularizer= l1(0.0000001), #l1_l2(0.0000001, 0.0000001), #l2(0.0000001),
                   return_sequences=True))
    model.add(LSTM(n_neurons, 
                   dropout = 0.5,
#                   recurrent_dropout = 0.3,
#                   kernel_regularizer=l1(0.0000001), #l1_l2(0.0000001, 0.0000001), 
#                   recurrent_regularizer=l1(0.0000001) #l1_l2(0.0000001, 0.0000001),
                   ))
    model.add(Dense(n_seq)) #, activation='tanh'  bias_constraint = NonNeg()
    model.compile(loss='mae', optimizer= Adam()) #default 0.001 learning_rate=0.008

	# fit network
    history = model.fit(train_X, train_y, epochs=nb_epoch, batch_size=n_batch, validation_data=(test_X, test_y), verbose=2, shuffle=False)
      
    # plot history for loss
    plt.plot(history.history['loss'], label='train') 
    plt.plot(history.history['val_loss'], label='test') 
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left') 
    plt.show()

    return model



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



# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):

	for i in range(n_seq):
		actual = [row[i] for row in test]
		predicted = [forecast[i] for forecast in forecasts]
		rmse = sqrt(mean_squared_error(actual, predicted))
		print('t+%d RMSE: %f' % ((i+1), rmse))

# plot the forecasts
def plot_forecasts(series, forecasts, n_test):

	# plot the entire dataset in blue
	plt.plot(series.values)

	# plot the forecasts in red
	for i in range(len(forecasts)):
		off_s = len(series) - n_test + i - 1
		off_e = off_s + len(forecasts[i]) + 1
		xaxis = [x for x in range(off_s, off_e)]
		yaxis = [series.values[off_s]] + forecasts[i]
		plt.plot(xaxis, yaxis, color='red')

	# show the plot
	plt.show()


    

# for visualization
def show_avg_diff_bw_fcst_actual(forecasts_avg, actual_inver):

    # evaluate: Avg Difference b/w last {n_test} value Forecast and Actual
    forecasts_sum = 0
    for i in range(len(forecasts_avg)): 
        forecasts_sum += forecasts_avg[i][-1]
    
    actual_sum = 0
    for i in range(len(actual_inver)): 
        actual_sum += actual_inver[i][-1]
        
    diff_avg = round((forecasts_sum - actual_sum) / n_test, 2)
        
    return print(f'Average Difference b/w the T+{n_seq} FCST and Actual: {diff_avg} in the last {n_test} days')


# for visualization
def get_df_for_linechart_fcst_actual(actual_inver, n_test, n_seq):

    # get actual T+{n_seq} fcst 
    l = []
    l_fo =[]
    for i in range(len(actual_inver)):
        actual = actual_inver[i][n_seq-1]
        fcst = forecasts_avg[i][n_seq-1]
        l.append(actual)
        l_fo.append(fcst)

    # get days for x-axis
    l_day = []
    for i in range(n_test):
        day = 'D{i}'.format(i=i+1)
        l_day.append(day)

    # make df
    dic = {"Actual": l, "Forecast": l_fo, "Day": l_day}
    df_acfo = pd.DataFrame(dic)
    df_acfo4plot = pd.melt(df_acfo, ["Day"])
    df_acfo4plot.columns = ['Day','Type','Qty']

    return df_acfo4plot


### START ###

df = df_raw.copy()


# model config
n_lag = 10 # input
n_seq = 1 # output
n_test = 10
n_epochs = 10
n_batch = 32
n_neurons = 8
n_features = df.shape[1]
n_obs = n_lag * df.shape[1]
n_runs = 1  # set n_runs > 1 if random seed not used at first  



# forecasting (跑n_runs取平均)
forecasts_all = []
for i in range(n_runs):
    print(f'Run: {i+1}')  
    # prepare data
    scaler, train, test = prepare_data(df, n_test, n_lag, n_seq)
    
    # split into input(x) output(y)
    train_X, train_y, test_X, test_y = split_into_input_output(train, test, n_seq)
    
    # fit model
    model = fit_lstm(train_X, train_y, test_X, test_y, n_lag, n_seq, n_batch, n_epochs, n_neurons)
    
    # make forecasts
    forecasts = make_forecasts(model, n_batch, test, n_lag, n_seq)
    
    # inverse transform forecasts and test
    forecasts_inver = inverse_transform(df, forecasts, scaler, n_test + (n_seq-1))
    forecasts_all.append(forecasts_inver)
    
    
# average of forecasts
forecasts_avg = list(np.mean(forecasts_all, axis=0))
forecasts_avg = [array.tolist() for array in forecasts_avg]

# actual values    
actual_col = [n_obs + n_features*i for i in range(n_seq)]
actual = [row[actual_col] for row in test]
actual_inver = inverse_transform(df, actual, scaler, n_test + (n_seq-1))
      

# evaluate forecasts 
evaluate_forecasts(actual_inver, forecasts_avg, n_lag, n_seq)

# plot actual/forecast
plot_forecasts(df.iloc[:,0], forecasts_avg, n_test + (n_seq-1))


## plot line chart comparing fcst/actual
df_acfo4plot = get_df_for_linechart_fcst_actual(actual_inver, n_test, n_seq)
p = sns.lineplot(x='Day', y='Qty', hue='Type', sort=False, data=df_acfo4plot)
p.tick_params(labelsize=6)
#plt.savefig('Z390 Sellin in Europe timeline.png', dpi=200)
plt.show() 


# show avgerage difference bw fcst actual
show_avg_diff_bw_fcst_actual(forecasts_avg, actual_inver)




### SAVE MODEL ###

# Save scaler 
#file_scaler = open('./model/scaler_20201214.pickle', 'wb')
#pickle.dump(scaler, file_scaler)
#file_scaler.close()

# Save model: creates a HDF5 file
#model.save('./model/LSTM_20201214.h5')

## from keras.models import load_model
#model = load_model('./API/data/LSTM_Z390_UA1_20200720.h5')
#
## reload scaler
#scaler = pickle.load(open('./API/data/LSTM_scaler_Z390_UA1_20200720.pickle', 'rb'))

    

