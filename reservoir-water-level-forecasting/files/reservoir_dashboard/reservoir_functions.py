import pandas as pd
import numpy as np

from datetime import datetime
from math import sqrt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow.keras

import pickle

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	import pandas as pd
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
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



def datasets(add = {}, data_source = 'wri', reservoir_name = 'KRS'):
  data_source = data_source.lower()
  if data_source == 'wri':
    df = pd.read_csv(add['WRI'], index_col=0)
    df = df[df['RESERVOIR']==reservoir_name]
    df = df.drop(["RESERVOIR"], axis = 1)
    df["PRESENT_STORAGE_TMC"] = pd.to_numeric(df["PRESENT_STORAGE_TMC"], errors='coerce')
    df = df.sort_values(by="DATE")
    df['DATE']=pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)
    df.drop(df.columns[[0,2,3,4,5]], axis=1, inplace=True)
    return df
    
  else:
    if data_source == 'indiawris':
      df = pd.read_csv(add['IndiaWRIS'], index_col=0)
      df = df[df['RESERVOIR']==reservoir_name]
      df = df.drop(["RESERVOIR"], axis = 1)
      df["PRESENT_STORAGE_TMC"] = pd.to_numeric(df["PRESENT_STORAGE_TMC"], errors='coerce')
      df = df.sort_values(by="DATE")
      df['DATE']=pd.to_datetime(df['DATE'])
      df.set_index('DATE', inplace=True)
      df.drop(df.columns[[0,1,3]], axis=1, inplace=True)
      return df
  print("Data Source not found.")
  return 

def pred_lstm_uni(add = {}, data_source = 'wri', reservoir_name = 'KRS', noofdays = 30):
  df = datasets(add, data_source, reservoir_name)

  values = df.values
  # ensure all data is float
  values = values.astype('float32')
  # normalize features
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled = scaler.fit_transform(values)
  # frame as supervised learning
  reframed = series_to_supervised(scaled, 1, 1) 
  # split into train and test sets
  values = reframed.values
  n_train_years = values.shape[0] - 366
  train = values[:n_train_years, :]
  test = values[n_train_years:, :]
  # split into input and outputs
  train_X, train_y = train[:, :-1], train[:, -1]
  test_X, test_y = test[:, :-1], test[:, -1]
  # reshape input to be 3D [samples, timesteps, features]
  train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
  test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
  # design network
  new_uni_model = Sequential()
  new_uni_model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
  new_uni_model.add(Dense(1))
  new_uni_model.compile(loss='mae', optimizer='adam')
  location = add['LSTM']
  # location = '/content/Wave2Web2021/fitted_models/lstm_uni_weights.h5'
  new_uni_model.load_weights(location)
  # make a prediction
  new_yhat = new_uni_model.predict(test_X)
  test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
  # invert scaling for forecast
  new_inv_yhat = np.concatenate((new_yhat, test_X[:, 1:]), axis=1)
  new_inv_yhat = scaler.inverse_transform(new_inv_yhat)
  new_inv_yhat = new_inv_yhat[:,0]
  # invert scaling for actual
  test_y = test_y.reshape((len(test_y), 1))
  new_inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
  new_inv_y = scaler.inverse_transform(new_inv_y)
  new_inv_y = new_inv_y[:,0]
  # calculate RMSE
  rmse = sqrt(mean_squared_error(new_inv_y[:90], new_inv_yhat[:90]))
  print('LSTM 90 days RMSE: %.3f' % rmse)
  rmse = sqrt(mean_squared_error(new_inv_y, new_inv_yhat))
  print('LSTM 1 year RMSE: %.3f' % rmse)

  pred_list = []
  n_input = 1
  n_features = 1
  batch_X = test_X[-n_input:].reshape((n_input, 1, n_features))
  n_pred = noofdays

  for i in range(n_pred):
    yhat = new_uni_model.predict(batch_X)
    batch = batch_X
    batch = batch.reshape(batch.shape[0], batch.shape[2])
    inv_yhat = np.concatenate((yhat, batch[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    pred_list.append(inv_yhat[-1])
    batch_X = np.append(batch_X, batch_X[0,0,0])
    batch_X = np.delete(batch_X,[0,0,0])
    batch_X[n_input-1] = yhat[-1]
    batch_X = batch_X.reshape((n_input, 1, n_features))
    
  add_dates = [df.index[-1] + pd.DateOffset(days=x) for x in range(0,n_pred+1) ]
  future_dates = pd.DataFrame(index=add_dates[1:],columns=df.columns)

  df_predict = pd.DataFrame(pred_list, index=future_dates[-n_pred:].index, columns=['FORECAST'])

  df_predict.index = pd.to_datetime(df_predict.index)
  df.index = pd.to_datetime(df.index)
  dfans = pd.concat([df, df_predict], axis=1, join="outer")
  return rmse, dfans

def pred_svr_uni(add = {}, data_source = 'wri', reservoir_name = 'KRS', noofdays = 30):
  df = datasets(add, data_source, reservoir_name)

  values = df.values
  # ensure all data is float
  values = values.astype('float32')
  # normalize features
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled = scaler.fit_transform(values)
  # frame as supervised learning
  reframed = series_to_supervised(scaled, 1, 1) 
  # split into train and test sets
  values = reframed.values
  n_test= 366
  X = reframed[['var1(t-1)']]
  y = reframed[['var1(t)']]
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = n_test, random_state=42, shuffle = False)
  filename = add['SVR']
  lm = pickle.load(open(filename, 'rb'))
  # make a prediction
  y_pred = lm.predict(X_test)
  y_pred = y_pred.reshape(-1,1)
  # invert scaling for forecast
  y_pred= scaler.inverse_transform(y_pred)
  # invert scaling for actual
  y_test=scaler.inverse_transform(y_test)
  # calculate RMSE
  rmse = sqrt(mean_squared_error(y_test[:90], y_pred[:90]))
  print('SVR 90 days RMSE: %.3f' % rmse)
  rmse = sqrt(mean_squared_error(y_test, y_pred))
  print('SVR 1 year RMSE: %.3f' % rmse)
  
  pred_list = []
  n_input = 1
  n_features = 1
  batch_X = X_test[-n_input:]
  n_pred = noofdays

  for i in range(n_pred):
    yhat = lm.predict(batch_X)
    batch_X = np.append(batch_X, yhat)
    batch_X = np.delete(batch_X,[0,0])
    batch_X = batch_X.reshape(-1,1)
    inv_yhat = scaler.inverse_transform(yhat.reshape(-1,1))
    pred_list.append(inv_yhat[-1])
    
  add_dates = [df.index[-1] + pd.DateOffset(days=x) for x in range(0,n_pred+1) ]
  future_dates = pd.DataFrame(index=add_dates[1:],columns=df.columns)

  df_predict = pd.DataFrame(pred_list, index=future_dates[-n_pred:].index, columns=['FORECAST'])
  
  df_predict.index = pd.to_datetime(df_predict.index)
  df.index = pd.to_datetime(df.index)
  dfans = pd.concat([df, df_predict], axis=1, join="outer")
  return rmse, dfans

def pred_xgb_uni(add = {}, data_source = 'wri', reservoir_name = 'KRS', noofdays = 30):
  df = datasets(add, data_source, reservoir_name)

  values = df.values
  # ensure all data is float
  values = values.astype('float32')
  # normalize features
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled = scaler.fit_transform(values.reshape(-1,1))
  # frame as supervised learning
  reframed = series_to_supervised(scaled, 1, 1) 
  # split into train and test sets
  values = reframed.values
  n_test= 366
  X = reframed[['var1(t-1)']]
  y = reframed[['var1(t)']]
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = n_test, random_state=42, shuffle = False)
  filename = add['XGBOOST']
  lm = pickle.load(open(filename, 'rb'))
  # make a prediction
  y_pred = lm.predict(X_test)
  y_pred = y_pred.reshape(-1,1)
  # invert scaling for forecast
  y_pred= scaler.inverse_transform(y_pred)
  # invert scaling for actual
  y_test=scaler.inverse_transform(y_test)
  # calculate RMSE
  rmse = sqrt(mean_squared_error(y_test[:90], y_pred[:90]))
  print('XGBoost 90 days RMSE: %.3f' % rmse)
  rmse = sqrt(mean_squared_error(y_test, y_pred))
  print('XGBoost 1 year RMSE: %.3f' % rmse)
  
  pred_list = []
  n_input = 1
  n_features = 1
  batch_X = X_test[-n_input:]
  n_pred = noofdays

  for i in range(n_pred):
    yhat = lm.predict(batch_X)
    batch_X = np.append(batch_X, yhat)
    batch_X = np.delete(batch_X,[0,0])
    batch_X = batch_X.reshape(-1,1)
    batch_X = pd.DataFrame(data=batch_X, columns=['var1(t-1)'])  
    inv_yhat = scaler.inverse_transform(yhat.reshape(-1,1))
    pred_list.append(inv_yhat[-1])
    
  add_dates = [df.index[-1] + pd.DateOffset(days=x) for x in range(0,n_pred+1) ]
  future_dates = pd.DataFrame(index=add_dates[1:],columns=df.columns)

  df_predict = pd.DataFrame(pred_list, index=future_dates[-n_pred:].index, columns=['FORECAST'])
  
  df_predict.index = pd.to_datetime(df_predict.index)
  df.index = pd.to_datetime(df.index)
  dfans = pd.concat([df, df_predict], axis=1, join="outer")
  return rmse,dfans

def forecast(add = {}, model_list = ['LSTM'], data_source='wri', reservoir_name='KRS', noofdays=90):  
  rmse = {}
  df = pd.DataFrame()
  if 'LSTM' in model_list:
    rmse['LSTM'], dflstm = pred_lstm_uni(add, data_source, reservoir_name, noofdays)
    dflstm['MODEL'] = 'LSTM'
    df = pd.concat([df,dflstm], axis=0, join="outer")
    dfensemblewt = dflstm.copy()
    dfensemble = dflstm.copy()

  if 'SVR' in model_list:
    rmse['SVR'], dfsvr = pred_svr_uni(add, data_source, reservoir_name, noofdays)
    dfsvr['MODEL'] = 'SVR'
    df = pd.concat([df,dfsvr], axis=0, join="outer")
    dfensemblewt = dfsvr.copy()
    dfensemble = dfsvr.copy()

  if 'XGBOOST'in model_list:
    rmse['XGBOOST'], dfxgb = pred_xgb_uni(add, data_source, reservoir_name, noofdays)
    dfxgb['MODEL'] = 'XGBOOST'
    df = pd.concat([df,dfxgb], axis=0, join="outer")
    dfensemblewt = dfxgb.copy()
    dfensemble = dfxgb.copy()

  dfensemblewt['MODEL'] = 'ENSEMBLE(WEIGHTED)'
  dfensemblewt['FORECAST'].values[:] = 0
  dfensemble['MODEL'] = 'ENSEMBLE(MEAN)'
  dfensemble['FORECAST'].values[:] = 0
  total = 0
  for model in model_list:
    wt = 0
    score = rmse[model]
    if score <=2:
      wt = 1
    elif score<=3:
      wt = 0.8
    elif score<=5:
      wt = 0.6
    elif score<=10:
      wt = 0.4
    elif score<=15:
      wt = 0.2
    else:
      wt = 0.1
    dfensemble['FORECAST'] += (df[df['MODEL']==model]['FORECAST'])
    dfensemblewt['FORECAST'] += (wt * df[df['MODEL']==model]['FORECAST'])
    total += wt
  dfensemble['FORECAST'] /= len(model_list)
  dfensemblewt['FORECAST'] /= total
  dfans = pd.concat([df, dfensemble], axis=0, join="outer")
  dfans = pd.concat([dfans, dfensemblewt], axis=0, join="outer")
  return rmse, dfans