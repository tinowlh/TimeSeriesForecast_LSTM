# model config
n_lag = 10 # input
n_seq = 1 # output
n_test = 10
n_epochs = 20
n_batch = 32
n_neurons = 8
n_runs = 1  # set n_runs > 1 if random seed not used at first 

# model saving
model_name = 'lstm_v1.0'
scaler_name = 'scaler_v1.0'

# data colloection
stockp_start_yr = 2018
stockp_start_mn = 12
stockp_start_d = 1

stockp_end_yr = 2020
stockp_end_mn = 12
stockp_end_d = 15

