#!/usr/bin/env python
# coding: utf-8

# Term Project - SARIMA Model
# Author: Faisal Hossain 2021

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# grid search sarima hyperparameters
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Read Datasets using Pandas
years = ["2014", "2015", "2016", "2017", "2018"]
year = "2018"
#for year in years:
df_zillow = pd.read_csv(f"../data/edit/zillow{year}_edit.csv", header = 0)

df_zillow['date'] = pd.to_datetime(df_zillow['date'])
y = df_zillow.set_index('date')

train = y[:int(0.75*(len(y)))]
valid = y[int(0.75*(len(y))):]
print(valid.index)

start_index = valid.index.min()
end_index = valid.index.max()

## SARIMA Model - Baseline
# Fit model
"""
model = SARIMAX(df_zillow, order=(3, 1, 3), seasonal_order=(1, 0, 2, 6))
model_fit = model.fit(disp=False)

start_index = valid.index(min(valid))

end_index = valid.index(max(valid))
"""

#building the model
from pmdarima import auto_arima
model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True, seasonal=True, m=6, stepwise=True)
model.fit(train)
# Predictions
pred = model.predict()
pred = model.predict(n_periods=len(valid))
pred = pd.DataFrame(pred, index = valid.index, columns=['Prediction'])

forecast = model.predict(n_periods=len(valid))
forecast = pd.DataFrame(forecast, index = valid.index, columns = ['Prediction'])

#plot the predictions for validation set
plt.plot(y.values, label='Train')
#plt.plot(valid, label='Valid')
plt.plot(forecast, label='Prediction')
plt.show()
"""
predictions = model_fit.predict(start=start_index, end=end_index)
print(yhat)

# Report performance
test = y[start_index:end_index]
mse = mean_squared_error(test, predictions)
rmse = sqrt(mse)
rsq = r2_score(test, predictions)
mape = mean_absolute_percentage_error(test, predictions)
print("MSE: %.3f" % mse)
print("RMSE: %.3f" % rmse)
print("Rsq: %.3f" % rsq)
print("MAPE: %.3f" % mape)

# Plot Model
plt.plot(y)
plt.plot(predictions)
plt.title(f"Zillow {year} SARIMA Model - Baseline")
plt.ylabel('Value $')
plt.xlabel('Date')
plt.show()

# Summary
print(model_fit.summary())

# Plot Stats
model_fit.plot_diagnostics(figsize=(18, 8))
plt.show()
"""
"""
rmse = sqrt(mean_squared_error(dataValues, yhat))
rsq = r2_score(dataValues, yhat)
print('Test RMSE: %.3f' % rmse)
print('Test Rsq: %.3f' % rsq)


pred = yhat.get_prediction(start=pd.to_datetime('1998-01-01'), dynamic=False)
pred_ci = pred.conf_int()

ax = df_zillow.values.plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Value $')
plt.legend()

plt.show()
"""

"""
## SARIMA Model - Grid Search
# one-step sarima forecast
def sarima_forecast(history, config):
	order, sorder, trend = config
	# define model
	model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
	# fit model
	model_fit = model.fit(disp=False)
	# make one step forecast
	yhat = model_fit.predict(len(history), len(history))
	return yhat[0]

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = sarima_forecast(history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
	return error

# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
	result = None
	# convert config to a key
	key = str(cfg)
	# show all warnings and fail on exception if debugging
	if debug:
		result = walk_forward_validation(data, n_test, cfg)
	else:
		# one failure during model validation suggests an unstable config
		try:
			# never show warnings when grid searching, too noisy
			with catch_warnings():
				filterwarnings("ignore")
				result = walk_forward_validation(data, n_test, cfg)
		except:
			error = None
	# check for an interesting result
	if result is not None:
		print(' > Model[%s] %.3f' % (key, result))
	return (key, result)

# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
	scores = None
	if parallel:
		# execute configs in parallel
		executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
		tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
		scores = executor(tasks)
	else:
		scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
	# remove empty results
	scores = [r for r in scores if r[1] != None]
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores

# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
	models = list()
	# define config lists
	p_params = [0, 1, 2]
	d_params = [0, 1]
	q_params = [0, 1, 2]
	t_params = ['n','c','t','ct']
	P_params = [0, 1, 2]
	D_params = [0, 1]
	Q_params = [0, 1, 2]
	m_params = seasonal
	# create config instances
	for p in p_params:
		for d in d_params:
			for q in q_params:
				for t in t_params:
					for P in P_params:
						for D in D_params:
							for Q in Q_params:
								for m in m_params:
									cfg = [(p,d,q), (P,D,Q,m), t]
									models.append(cfg)
	return models
data = list(df_zillow.values.flatten())
# data split
n_test = 4
# model configs
cfg_list = sarima_configs()
# grid search
scores = grid_search(data, cfg_list, n_test)
print('done')
# list top 3 configs
for cfg, error in scores[:3]:
        print(cfg, error)
"""
