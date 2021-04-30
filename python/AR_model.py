#!/usr/bin/env python
# coding: utf-8

# Term Project
# Author: Faisal Hossain 2021

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot, autocorrelation_plot
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from math import sqrt

# Read Datasets using Pandas
years = ["2014", "2015", "2016", "2017", "2018"]
years = ["2018"]
for year in years:
        df_zillow = pd.read_csv(f"../data/edit/zillow{year}_edit.csv", header = 0, index_col=0)
        #df_zillow = pd.read_csv('../data/zillow_2014.csv', header = 0, index_col=0)

        # Plot
        #df_zillow.drop(['region_id'], axis=1).plot()
        df_zillow.plot()
        plt.title(f"Zillow {year} Timeseries Model")
        plt.ylabel('Value $')
        plt.xlabel('Date')
        plt.show()

        # Lag Plot
        lag_plot(df_zillow)
        plt.title(f"Zillow {year} Lag Plot")
        plt.show()

        # Autocorrelation Coefficient
        values = df_zillow.value
        dataframe = pd.concat([values.shift(1), values], axis=1)
        dataframe.columns = ['t-1', 't+1']
        print(dataframe.corr())

        # Autocorrelation Plot1
        autocorrelation_plot(df_zillow)
        plt.title(f"Zillow {year} Autocorrelation Plot")
        plt.ylabel('Correlation Coefficient')
        plt.xlabel('Lag')
        plt.show()

        # Autocorrelation Plot2
        plot_acf(df_zillow, lags=31)
        plt.title(f"Zillow {year} Autocorrelation Plot")
        plt.ylabel('Correlation Coefficient')
        plt.xlabel('Lag')
        plt.show()
        
        ## Autoregression Model - Regular
        # Split dataset
        days = [20, 365]
        for daysNum in days:
                X = df_zillow.values
                print(str(daysNum) + "Forecast")
                train, test = X[1:len(X)-daysNum], X[len(X)-daysNum:]

                # Train model
                model = AutoReg(train, lags=29)
                model_fit = model.fit()
                print('Coefficients: %s' % model_fit.params)

                # Make predictions
                print()
                predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

                for i in range(len(predictions)):
                        """
                        print(f"expected: {test[i]}, predicted: {predictions[i]}")
                        absoluteError = abs(test[i] - predictions[i])
                        relativeError = absoluteError / test[i]
                        precentError = str(relativeError * 100) + '%'
                        print(f"difference: {absoluteError}, 'relative error: {relativeError}, percent error: {precentError}")
                        print()
                        """
                rmse = sqrt(mean_squared_error(test, predictions))
                rsq = r2_score(test, predictions)
                mape = mean_absolute_percentage_error(test, predictions)
                print("RMSE: %.3f" % rmse)
                print("Rsq: %.3f" % rsq)
                print(f"MAPE: {mape}")

                # Plot results
                plt.plot(test)
                plt.plot(predictions, color='red')
                plt.title(f"Zillow {year} AR Model- Baseline {daysNum}")
                plt.ylabel('Value $')
                plt.xlabel('Date')
                plt.show()
        

        ## Autoregression Model - Manual
        # Use the learned coefficients and manually make predictions

        # Split dataset
        X = df_zillow.values
        train, test = X[1:len(X)-365], X[len(X)-365:]

        # Train model
        window = 29
        model = AutoReg(train, lags=29)
        model_fit = model.fit()
        coef = model_fit.params

        # Walk forward over time steps in test
        history = train[len(train)-window:]
        history = [history[i] for i in range(len(history))]
        predictions = list()
        for t in range(len(test)):
                length = len(history)
                lag = [history[i] for i in range(length-window,length)]
                yhat = coef[0]
                for d in range(window):
                        yhat += coef[d+1] * lag[window-d-1]
                obs = test[t]
                predictions.append(yhat)
                history.append(obs)
                """
                print(f"expected: {obs}, predicted: {yhat}")
                absoluteError = abs(obs - yhat)
                relativeError = absoluteError / obs
                precentError = str(relativeError * 100) + '%'
                print(f"difference: {absoluteError}, 'relative error: {relativeError}, percent error: {precentError}")
                print()
                """
                
        rmse = sqrt(mean_squared_error(test, predictions))
        rsq = r2_score(test, predictions)
        mape = mean_absolute_percentage_error(test, predictions)
        print(f"RMSE: {rmse}")
        print("Rsq: %.3f" % rsq)
        print(f"MAPE: {mape}")
        
        # Plot results
        plt.plot(test,label='actual')
        plt.plot(predictions,label='predicted')
        plt.title(f"Zillow {year} AR Model")
        plt.ylabel('Value $')
        plt.xlabel('Date')
        plt.legend()
        plt.show()
