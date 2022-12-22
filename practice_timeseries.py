
# https://levelup.gitconnected.com/simple-forecasting-with-auto-arima-python-a3f651271965

# https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/

# Calculate the first difference and drop the nans
from statsmodels.tsa.stattools import adfuller
# Import augmented dicky-fuller test function
# from statsmodels.tsa.stattools import adfuller
#
# Import modules
import pandas as pd
import matplotlib.pyplot as plt

# Load in the time series
candy = pd.read_csv('candy_production.csv',
                 index_col='date',
                 parse_dates=True)

# Plot and show the time series on axis ax1
fig, ax1 = plt.subplots()
candy.plot(ax=ax1)
plt.show()
# candy_train = candy.loc[:'2006']
# candy_test = candy.loc['2007':]
#
# # Create an axis
# fig, ax = ____
#
# # Plot the train and test sets on the axis ax
# candy_train.plt.subplots(ax=ax1)
# candy_test.plt.subplot(ax=ax1)
# plt.show()

# # Run test
# result = adfuller(earthquake['earthquakes_per_year'])
#
# # Print test statistic
# print(result[0])
#
# # Print p-value
# print(result[1])
#
# # Print critical values
# print(result[4])




# # Run the ADF test on the time series
# result = adfuller(city['city_population'])
#
# # Plot the time series
# fig, ax = plt.subplots()
# city.plot(ax=ax)
# plt.show()

# Print the test statistic and the p-value
# print('ADF Statistic:', result[0])
# print('p-value:', result[1])
#
#
#
# amazon_diff = amazon.diff()
# amazon_diff = amazon_diff.dropna()

# Run test and print
# result_diff = adfuller(amazon_diff['close'])
# print(result_diff)
# print(result_diff)
import numpy as np
# Import data generation function and set random seed
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import arma_generate_sample
np.random.seed(1)

# Set coefficients
ar_coefs = [1]
ma_coefs = [1, -0.7]

# Generate data
y = arma_generate_sample(ar_coefs, ma_coefs, nsample=100, scale=0.5)

plt.plot(y)
plt.ylabel(r'$y_t$')
plt.xlabel(r'$t$')
plt.show()

# Import the ARIMA model
from statsmodels.tsa.arima.model import ARIMA

# Instantiate the model
model = ARIMA(y, order=(1,0,1))

# Fit the model
results = model.fit()
print(results)