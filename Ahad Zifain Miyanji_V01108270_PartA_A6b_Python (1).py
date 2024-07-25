#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model


# In[2]:


# Load the dataset
data = pd.read_csv("NFLX Historical Data.csv")
data.head()


# In[3]:


print(data.info())


# In[4]:


# Check for missing values and handle them
print(data.isnull().sum())


# In[5]:


# Interpolate or remove missing values
# data = data.interpolate()  # Interpolate missing values
# data = data.dropna()  # Drop any remaining missing values


# In[6]:


# Convert string values to float
data['Vol.'] = data['Vol.'].str.replace('M', '').astype(float) * 1e6  # Convert volume to float and handle 'M' for millions
data['Change %'] = data['Change %'].str.replace('%', '').astype(float)  # Convert change percentage to float


# In[7]:


# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'])


# In[8]:


# Check for ARCH/GARCH effects
# For this, we can use the squared returns (log returns)
data['log_return'] = np.log(data['Price']).diff()
data = data.dropna()  # Drop NaN values created by differencing
data['squared_log_return'] = data['log_return'] ** 2


# In[9]:


# Plot squared log returns to visually check for ARCH effects
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['squared_log_return'])
plt.title('Squared Log Returns')
plt.show()


# In[10]:


# Fit an ARCH/GARCH model
# We'll use a simple GARCH(1, 1) model for this example
model = arch_model(data['log_return'], vol='Garch', p=1, q=1)
model_fit = model.fit(disp='off')
print(model_fit.summary())


# In[11]:


# Forecasting the three-month volatility
forecast_horizon = 3 * 30  # Approximate days for three months
forecasts = model_fit.forecast(horizon=forecast_horizon)


# In[12]:


# Extract the forecasted variances and convert to volatility (standard deviation)
forecasted_volatility = np.sqrt(forecasts.variance.values[-1])


# In[13]:


# Create a DataFrame to display the forecasted values
forecast_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_horizon)
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted_Volatility': forecasted_volatility})


# In[14]:


# Display the forecasted values
print(forecast_df)


# In[15]:


# Plot the forecasted volatility
plt.figure(figsize=(10, 6))
plt.plot(forecast_df['Date'], forecast_df['Forecasted_Volatility'])
plt.title('Three-Month Volatility Forecast')
plt.xlabel('Date')
plt.ylabel('Forecasted Volatility')
plt.show()


# ### Part B : VAR/VECM

# In[1]:


# Import necessary libraries
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM


# In[2]:


# Load the dataset
df = pd.read_excel('pinksheet.xlsx', sheet_name="Monthly Prices", skiprows=6)


# In[3]:


# Display the first few rows of the dataset
print(df.head())


# In[5]:


# Rename the first column to "Date"
df.rename(columns={df.columns[0]: 'Date'}, inplace=True)


# In[6]:


# Check for missing values
print(df.isnull().sum())


# In[7]:


# Fill or drop missing values as needed
# Here, we'll use forward fill method
df.fillna(method='ffill', inplace=True)


# In[16]:


# Select relevant columns (example columns based on your task description)
# Update the column names as per your dataset
columns = ['CRUDE_PETRO', 'CRUDE_BRENT', 'CRUDE_DUBAI','SUGAR_EU', 'SUGAR_US','GOLD', 'PLATINUM', 'SILVER']


# In[17]:


data = df[columns]


# In[11]:


# Strip any leading/trailing spaces from column names
# df.columns = df.columns.str.strip()


# In[18]:


# Display the cleaned column names
# print("DataFrame Columns:")
# print(df.columns)


# In[19]:


# Update the list of columns to match the actual column names
# columns = ['Crude oil, average', 'Crude oil, Brent', 'Crude oil, Dubai', 'Crude oil, WTI', 
#            'Coal, Australian', 'Coal, South African **', 'Natural gas, US', 
#            'Natural gas, Europe', 'Liquefied natural gas, Japan', 'Natural gas index']


# In[20]:


# Display the list of columns
# print("Expected Columns:")
# print(columns)


# In[21]:


# Check if all columns are present in the DataFrame
# missing_columns = [col for col in columns if col not in df.columns]
# if missing_columns:
#     print(f"Missing columns: {missing_columns}")

#     # Investigate why the columns are missing
#     for col in columns:
#         if col not in df.columns:
#             print(f"Column '{col}' is missing from DataFrame columns.")
# else:
#     print("All columns are present in the DataFrame.")

#     # Select relevant columns
#     data = df[columns]

#     # Ensure all data is numeric
#     data = data.apply(pd.to_numeric)

#     # Display the first few rows of the selected data
#     print(data.head())


# In[22]:


# Ensure all data is numeric
data = data.apply(pd.to_numeric)


# In[23]:


# VAR Model
model_var = VAR(data)
results_var = model_var.fit(maxlags=15, ic='aic')


# In[24]:


# Print summary of VAR model results
print(results_var.summary())


# In[28]:


# Forecast with VAR model
n_forecast = 10  # Number of steps to forecast
forecast_var = results_var.forecast(data.values[-results_var.k_ar:], steps=n_forecast)
forecast_var_df = pd.DataFrame(forecast_var, index=pd.date_range(start=data.index[-1], periods=n_forecast+1, freq='M')[1:], columns=data.columns)
print("VAR Model Forecast:")
print(forecast_var_df)


# In[25]:


# VECM Model
# Perform Johansen cointegration test to determine the number of cointegrating relationships
johansen_test = coint_johansen(data, det_order=0, k_ar_diff=1)
print(johansen_test.lr1)  # Trace statistic
print(johansen_test.cvt)  # Critical values


# In[26]:


# Assuming one cointegrating relationship for VECM
model_vecm = VECM(data, k_ar_diff=1, coint_rank=1)
results_vecm = model_vecm.fit()


# In[27]:


# Print summary of VECM model results
print(results_vecm.summary())


# In[29]:


# Forecast with VECM model
forecast_vecm = results_vecm.predict(steps=n_forecast)
forecast_vecm_df = pd.DataFrame(forecast_vecm, index=pd.date_range(start=data.index[-1], periods=n_forecast+1, freq='M')[1:], columns=data.columns)
print("VECM Model Forecast:")
print(forecast_vecm_df)


# In[ ]:




