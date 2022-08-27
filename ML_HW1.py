#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AutoReg
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
import copy


# In[3]:


BSP = pd.read_csv('Brent Spot Price.csv')
BSP.rename(
    columns={"Unnamed: 0":"date",
                "Brent crude oil spot price, Monthly (dollars per barrel)":"price"}
          ,inplace=True)
BSP['date'] = pd.to_datetime(BSP['date'])
BSP.set_index('date', inplace = True)
print(BSP.head())
BSP.info()


# In[4]:


CP = pd.read_csv('Coal Power.csv')
CP.rename(
    columns={"Unnamed: 0":"date",
                "Total consumption : Texas : electric power (total) : quarterly (short tons)":"power"}
          ,inplace=True)
CP["date"] = CP["date"].str.replace("Q1","01") 
CP["date"] = CP["date"].str.replace("Q2","04") 
CP["date"] = CP["date"].str.replace("Q3","07") 
CP["date"] = CP["date"].str.replace("Q4","10") 
print(CP.head(10))
CP['date'] = pd.to_datetime(CP['date'], utc=False)
CP.set_index('date', inplace = True)
print(CP.head())
CP.info()


# In[4]:


BSP.plot()


# In[5]:


CP.plot()


# In[6]:


result_BSP= adfuller(BSP)
print('BSP', '\n',result_BSP)
result_CP= adfuller(CP)
print('CP', '\n',result_CP)


# # differencing  

# In[7]:


bsp_copy = copy.deepcopy(BSP)
cp_copy = copy.deepcopy(CP)
price_diff = np.diff(bsp_copy.price, n=1)
power_diff = np.diff(cp_copy.power, n=2)


# In[8]:


result_BSP_diff= adfuller(price_diff)
print('BSP with differencing', '\n',result_BSP_diff)
result_CP_diff= adfuller(power_diff)
print('CP with differencing', '\n',result_CP_diff)


# In[9]:


BSP_diff = bsp_copy.iloc[1: , :]
BSP_diff['price'] = price_diff
CP_diff = cp_copy.iloc[2:, :]
CP_diff['power'] = power_diff


# # Moving averrage

# In[6]:


for i in range(1,50):
  CPbest = CP['power'].rolling(window=i).mean()
  CPbest.dropna(inplace = True)
  CPbest_dataframe = pd.DataFrame(CPbest)
  CPbest_values = CPbest_dataframe.iloc[:,0].values
  ADF_result_BSP_moving_windowbest  = adfuller(CPbest_values)
  print('ADF_result_BSP_moving_window',i,":",ADF_result_BSP_moving_windowbest)

CP23 = CP['power'].rolling(window=23).mean()
CP30 = CP['power'].rolling(window=30).mean()
CP10 = CP['power'].rolling(window=10).mean()

CP23.dropna(inplace = True)
CP23_dataframe = pd.DataFrame(CP23)
CP23_values = CP23_dataframe.iloc[:,0].values
ADF_result_BSP_moving_window23 = adfuller(CP23_values)
print('ADF_result_CP_moving_window23',ADF_result_BSP_moving_window23 )

CP30.dropna(inplace = True)
CP30_dataframe = pd.DataFrame(CP30)
CP30_values = CP30_dataframe.iloc[:,0].values
ADF_result_BSP_moving_window30  = adfuller(CP30_values)
print('ADF_result_CP_moving_window30',ADF_result_BSP_moving_window30)

CP10.dropna(inplace = True)
CP10_dataframe = pd.DataFrame(CP10)
CP10_values = CP10_dataframe.iloc[:,0].values
ADF_result_BSP_moving_window10  = adfuller(CP10_values)
print('ADF_result_CP_moving_window10',ADF_result_BSP_moving_window10)


plt.title('coal power moving averages', size=20)
plt.plot(CP['power'], label='Original')
plt.plot(CP23, color='gray', label='50')
plt.plot(CP30, color='orange', label='30')
plt.plot(CP10, color='red', label='10')


# In[5]:


for i in range(1,100):
  BSPbest = BSP['price'].rolling(window=i).mean()
  BSPbest.dropna(inplace = True)
  BSPbest_dataframe = pd.DataFrame(BSPbest)
  BSPbest_values = BSPbest_dataframe.iloc[:,0].values
  ADF_result_BSP_moving_windowbest  = adfuller(BSPbest_values)
  print('ADF_result_BSP_moving_window',i,":",ADF_result_BSP_moving_windowbest)

BSP50 = BSP['price'].rolling(window=50).mean()
BSP30 = BSP['price'].rolling(window=30).mean()
BSP100 = BSP['price'].rolling(window=100).mean()
BSP50.dropna(inplace = True)
BSP50_dataframe = pd.DataFrame(BSP50)
BSP50_dataframe_values = BSP50_dataframe.iloc[:,0].values
ADF_result_BSP_moving_window50 = adfuller(BSP50_dataframe_values)
print('ADF_result_BSP_moving_window50',ADF_result_BSP_moving_window50)

BSP30.dropna(inplace = True)
BSP30_dataframe = pd.DataFrame(BSP30)
BSP30_dataframe_values = BSP30_dataframe.iloc[:,0].values
ADF_result_BSP_moving_window30 = adfuller(BSP30_dataframe_values)
print('ADF_result_BSP_moving_window30',ADF_result_BSP_moving_window30)

BSP100.dropna(inplace = True)
BSP100_dataframe = pd.DataFrame(BSP100)
BSP100_dataframe_values = BSP100_dataframe.iloc[:,0].values
ADF_result_BSP_moving_window100 = adfuller(BSP30_dataframe_values)
print('ADF_result_BSP_moving_window100',ADF_result_BSP_moving_window100)


plt.title('price of Brent Spot Price moving averages', size=20)
plt.plot(BSP['price'], label='Original')
plt.plot(BSP50, color='gray', label='50')
plt.plot(BSP30, color='orange', label='30')
plt.plot(BSP100, color='red', label='100')


# # Decomposition 

# In[ ]:





# In[12]:


decompose_bsp = seasonal_decompose(BSP_diff, model = 'additive')
decompose_bsp.plot()


# In[13]:


decompose_cp = seasonal_decompose(CP_diff, model = 'additive', period = int(len(CP_diff)/2))
decompose_cp.plot()


# # Autocorrelation

# In[14]:


sm.graphics.tsa.plot_acf(BSP_diff ,lags=30)
plt.show()


# In[15]:


sm.graphics.tsa.plot_acf(CP_diff ,lags=30)
plt.show()


# # Partial autocorrelation

# In[16]:


sm.graphics.tsa.plot_pacf(BSP_diff ,lags=30, method="ywm")
plt.show()


# In[17]:


sm.graphics.tsa.plot_pacf(CP_diff ,lags=30, method="ywm")
plt.show()


# # Split train and test data

# In[18]:


#BSP
train_size = int(len(BSP_diff)*0.8)
bsp_train = BSP_diff.iloc[:train_size, :]
bsp_test = BSP_diff.iloc[train_size:, :]


# In[19]:


#cp
train_size_cp = int(len(CP_diff)*0.8)
cp_train = CP_diff.iloc[:train_size_cp, :]
cp_test = CP_diff.iloc[train_size_cp:, :]


# In[20]:


print(len(cp_train), len(cp_test))


# # AR 

# In[21]:


#BSP
ar_bsp = AutoReg(bsp_train, lags =6).fit()
print (ar_bsp.summary())


# In[22]:


#
# Make the predictions
#
bsp_pred = ar_bsp.predict(start=len(bsp_train), end=(len(BSP_diff)-1), dynamic=False)

plt.plot(bsp_pred)
plt.plot(bsp_test, color='red')


# In[23]:


ar_cp = AutoReg(cp_train, lags =3).fit()
print (ar_cp.summary())


# In[24]:


#
# Make the predictions
#
cp_pred = ar_cp.predict(start=len(cp_train), end=(len(CP_diff)-1), dynamic=False)
plt.figure()
plt.plot(cp_pred)
plt.plot(cp_test, color='red')


# # MA

# In[25]:


ma_bsp = ARIMA(bsp_train, order=(0,0,6)).fit()
print(ma_bsp.summary())


# In[26]:


bsp_pred = ma_bsp.predict(start=len(bsp_train), end=(len(BSP_diff)-1), dynamic=False)

plt.plot(bsp_pred)
plt.plot(bsp_test, color='red')


# In[27]:


ma_cp = ARIMA(cp_train, order=(0,0,3)).fit()
print(ma_cp.summary())


# In[28]:


cp_pred = ma_cp.predict(start=len(cp_train), end=(len(CP_diff)-1), dynamic=False)
plt.figure()
plt.plot(cp_pred)
plt.plot(cp_test, color='red')


# In[ ]:





# # ARMA

# In[29]:


arma_bsp = ARIMA(bsp_train, order=(2,0,3)).fit()
print(arma_bsp.summary())


# In[30]:


bsp_pred = arma_bsp.predict(start=len(bsp_train), end=(len(BSP_diff)-1), dynamic=False)

plt.plot(bsp_pred)
plt.plot(bsp_test, color='red')


# In[31]:


arma_cp = ARIMA(cp_train, order=(2,0,1)).fit()
print(arma_cp.summary())


# In[32]:


cp_pred = arma_cp.predict(start=len(cp_train), end=(len(CP_diff)-1), dynamic=False)
plt.figure()
plt.plot(cp_pred)
plt.plot(cp_test, color='red')


# # ARIMA

# In[33]:


#BSP
train_size_arima = int(len(BSP)*0.8)
bsp_train_arima = BSP.iloc[:train_size_arima, :]
bsp_test_arima = BSP.iloc[train_size_arima:, :]


# In[34]:


#cp
train_size_cp_arima = int(len(CP)*0.8)
cp_train_arima = CP.iloc[:train_size_cp_arima, :]
cp_test_arima = CP.iloc[train_size_cp_arima:, :]


# In[35]:


arima_bsp = ARIMA(bsp_train_arima, order=(2,1,3)).fit()
print(arima_bsp.summary())


# In[36]:


bsp_pred = arima_bsp.predict(start=len(bsp_train_arima), end=(len(BSP_diff)-1), dynamic=False)

plt.plot(bsp_pred)
plt.plot(bsp_test_arima, color='red')


# In[37]:


arima_cp = ARIMA(cp_train_arima, order=(2,1,1)).fit()
print(arima_cp.summary())


# In[38]:


cp_pred = arima_cp.predict(start=len(cp_train_arima), end=(len(CP_diff)-1), dynamic=False)
plt.figure()
plt.plot(cp_pred)
plt.plot(cp_test_arima, color='red')


# In[ ]:




