# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
get_ipython().run_line_magic('matplotlib', 'inline')

# For reading stock data from yahoo
from pandas_datareader.data import DataReader

# For time stamps
from datetime import datetime


# %%
#we are analyzing the stock Apple google Microsoft, Amazon
tech_list =['AAPL', 'GOOG', 'MSFT', 'AMZN']

#Set up End and Start times for data Grabs
end = datetime.now()
start = datetime(end.year -1, end.month, end.day)

#for loop for grabing yahoo finance data and seting as a data frame
for stock in tech_list:
    #set dataframe as stock ticker
    globals()[stock] = DataReader(stock, 'yahoo', start,end)


# %%
company_list = [AAPL, GOOG, MSFT, AMZN] 
company_name = ['APPLE', 'GOOGLE','MICROSOFT', 'AMAZON']

for company, company_name in zip(company_list, tech_list):
    company['company_name'] = company_name

df = pd.concat(company_list, axis=0)
df.tail(10)


# %%
AAPL.describe()
AAPL.info()


# %%
#let see the histrocial view of the  closing proce

plt.figure(figsize=(12,8))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, company in enumerate(company_list,1):
    plt.subplot(2,2,i)
    company['Adj Close'].plot()
    plt.ylabel('Adj Close')
    plt.xlabel(None)
    plt.title(f"{tech_list[i-1]}")


# %%
## now the plot the total Volume of stock being traded each day

plt.figure(figsize =(12,8))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, company in enumerate(company_list,1):
    plt.subplot(2,2,i)
    company['Volume'].plot()
    plt.ylabel('Volume')
    plt.xlabel(None)
    plt.title(f"{tech_list[i-1]}")
    


# %%
##2. What was the moving Average of the stock 
ma_day =[10,20,50]

for ma in ma_day:
    for company in company_list:
        column_name = f"MA for {ma} days"
        company[column_name] = company['Adj Close'].rolling(ma).mean()
print(AMZN.columns)


# %%

#Grab all the moving Average
df.groupby("company_name").hist(figsize=(12,12));


# %%

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(8)
fig.set_figwidth(15)

AAPL[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0,0])
axes[0,0].set_title('APPLE')

GOOG[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0,1])
axes[0,1].set_title('GOOGLE')

MSFT[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1,0])
axes[1,0].set_title('MICROSOFT')

AMZN[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1,1])
axes[1,1].set_title('AMAZON')


fig.tight_layout()


# %%
#Daily return on the stock 
# We'll use the pct_change to find percentage change for each day 


# %%
for company in company_list:
    company['Daily Return'] = company['Adj Close'].pct_change()

# plot the daily return percentage
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(8)
fig.set_figwidth(15)

AAPL['Daily Return'].plot(ax=axes[0,0], legend=True, linestyle='--', marker='o')
axes[0,0].set_title('APPLE')

GOOG['Daily Return'].plot(ax=axes[0,1], legend=True, linestyle='--', marker='o')
axes[0,1].set_title('GOOGLE')

MSFT['Daily Return'].plot(ax=axes[1,0], legend=True, linestyle='--', marker='o')
axes[1,0].set_title('MICROSOFT')

AMZN['Daily Return'].plot(ax=axes[1,1], legend=True, linestyle='--', marker='o')
axes[1,1].set_title('AMAZON')

fig.tight_layout()
#use the drop NAN and draw daily return of the stock 


# %%
plt.figure(figsize=(12,12))

for i, company in enumerate(company_list, 1):
     plt.subplot(2,2,i)
     sns.distplot(company['Daily Return'].dropna(), bins=100, color='purple')
     plt.ylabel('Daily Return')
     plt.title(f'{company_name[i-1]}')


# %%
#4. what is the correlation between differnet stocks closing prices
closing_df = DataReader(tech_list, 'yahoo', start,end)['Adj Close']
#take a quick look at the closing df
closing_df.head(10)


# %%
#make a new tech returns DataFrame
tech_rets = closing_df.pct_change()
tech_rets.head()


# %%
#compare return of Google and Microsoft. Obviously, two stocks are perfectly correlated with each other a linear relationship between its daily return values should occur.
sns.jointplot('GOOG','MSFT', tech_rets, kind='scatter')
#


# %%
#use pairplot on our DataFrame for an automatic visual analysis
sns.pairplot(tech_rets, kind='reg')

#%%
#The correlation can be done by using Seaborn sns.pairplot to examine the correlation of invidual stocks

#Define return_figure using PairGrid
return_fig = sns.PairGrid(tech_rets.dropna())

#Using map_upper we can specify what upper triangle will look like.
return_fig.map_upper(plt.scatter, color='purple')

#we can also define the lower triagle in the figure
#Color map()
return_fig.map_lower(sns.kdeplot,cmap='cool_d')

#define the diagonal as a series of histrogram plot of the daily return
return_fig.map_diag(plt.hist, bins=30)


#%%
#Define return_figure using PairGrid
return_fig = sns.PairGrid(closing_df)

#Using map_upper we can specify what upper triangle will look like.
return_fig.map_upper(plt.scatter, color='purple')

#we can also define the lower triagle in the figure
#Colo map()
return_fig.map_lower(sns.kdeplot,cmap='cool_d')

#define the diagonal as a series of histrogram plot of the daily return
return_fig.map_diag(plt.hist, bins=30)


# %%
# Quick corellation on plot for the daily returns
sns.heatmap(tech_rets.corr(), annot=True,cmap='summer')
#%%

sns.heatmap(closing_df.corr(), annot=True,cmap='summer')
# %%
#New DataFrame withou the version of origianl Tech_rets DataFrame

rets = tech_rets.dropna()

area = np.pi*20

plt.figure(figsize=(12, 10))
plt.scatter(rets.mean(), rets.std(), s=area)
plt.xlabel('Expected return')
plt.ylabel('Risk')

for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(label, xy=(x,y), xytext=(50, 50), textcoords='offset points', ha='right', va='bottom', 
                 arrowprops=dict(arrowstyle='-', color='blue', connectionstyle='arc3,rad=-0.3'))

# %%
#6. Predict the stocK LEMONDADE
#Get the stock quote
df = DataReader('AAPL', data_source='yahoo', start='2012-01-01', end=datetime.now())
#Show the price over time
df
# %%
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()

#%%Create the new Dataframe with only Close column
data=df.filter(['Close'])
#convert the dataframe to the numpy array
dataset=data.values
#Get the number of rows to train the model on 
training_data_len = int(np.ceil(len(dataset) * 0.8))
training_data_len
# %%
#Scale the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data
# %%
#Create the training data set
#Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]
#Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()
        
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# x_train.shape
#%%
#Build the LSTM model
from keras.models import Sequential
from keras.layers import Dense, LSTM

#Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences= False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)
#%%
#Create the testing data set
#Create a new array containing scaled values from index 1543 to 2002 
test_data = scaled_data[training_data_len - 60: , :]
#Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
rmse
#%%
data

#%%
#plot the data 
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions']= predictions
#Visualize the data 
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Data, fontsize=18')
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
# %%
#Show the valid and predicted prices
valid