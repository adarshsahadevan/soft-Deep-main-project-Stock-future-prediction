import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import yfinance as yf

st.title("Stock Future Prediction App")

st.session_state.data = None
stock = st.text_input("ENTER THE STOCK ID    (You can get your favorite Stock's ID from Yahoo Finance.)" , "GOOG")

from datetime import datetime
end = datetime.now()
start = datetime(end.year - 15, end.month, end.day)

df = yf.download(stock, start, end)

model = load_model("stock_future_prediction_saved.keras")

st.subheader("Stock Data : ")
st.subheader("Head of the stock data : ")
st.write(df.head())
st.subheader("Tail of the stock data : ")
st.write(df.tail())

st.subheader("Closing Price of the Stock : ")
plt.figure(figsize = (15,5))
plt.plot(df['Close'])
plt.xlabel("Year")
plt.ylabel("Close Price")
plt.title("CLOSING PRICE")
plt.grid(True)
st.pyplot(plt)

ma100 = df.Close.rolling(100).mean()
st.subheader("100 days Moving Average :")
plt.figure(figsize = (15,5))
plt.plot(df.Close, label='Close price')
plt.plot(ma100,'r', label= 'Moving Average 100 days')
plt.legend()
plt.grid(True)
st.pyplot(plt)

ma200 = df.Close.rolling(200).mean()
st.subheader("200 days Moving Average :")
plt.figure(figsize = (15,5))
plt.plot(df.Close, label='Close price')
plt.plot(ma100,'r', label= 'Moving Average 100 days')
plt.plot(ma200,'g', label= 'Moving Average 200 days')
plt.legend()
plt.grid(True)
st.pyplot(plt)

Close_price = df['Close']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform(Close_price.values.reshape(-1,1))

x_data = []
y_data = []

for i in range(100,len(scaled_data)):
  x_data.append(scaled_data[i-100:i])
  y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data) , np.array(y_data)

splitting_len = int(len(x_data)*0.7)
x_train = x_data[:splitting_len]
y_train = y_data[:splitting_len]

x_test = x_data[splitting_len:]
y_test = y_data[splitting_len:]

from keras.layers import Dense, LSTM
from keras.models import Sequential

model = Sequential()
model.add(LSTM(units = 128, return_sequences =True, input_shape = (x_train.shape[1], 1 )))
model.add(LSTM(units = 64, return_sequences =False))

model.add(Dense(units = 25))
model.add(Dense(units = 1))

model.compile(optimizer = 'adam' , loss = 'mean_squared_error')
model.fit(x_train, y_train, epochs = 5)

predictions = model.predict(x_test)

predictions = predictions.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

inv_predictions = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_test)

ploting_data = pd.DataFrame(
    {
     "Date": df.index[splitting_len+100:],
     "Actual": inv_y_test.reshape(-1),
     "Predicted":inv_predictions.reshape(-1)[:len(inv_y_test)]
    },
)
st.subheader("Original values vs predicted values : ")
st.write(ploting_data)

st.subheader("Original Close price vs predicted Close price : ")


ploting_data['Date'] = pd.to_datetime(ploting_data['Date'])
unique_years = ploting_data['Date'].dt.year.unique()


plt.figure(figsize=(15, 5))
plt.plot(ploting_data['Date'], ploting_data['Actual'], label='Actual')
plt.plot(ploting_data['Date'], ploting_data['Predicted'], label='Predicted')
plt.xlabel("Year")
plt.ylabel("Closing Price Actual & Predicted")
plt.title("Tested Data")
plt.xticks(ticks=pd.to_datetime(unique_years.astype(str)), labels=unique_years)
plt.legend()
plt.tight_layout()
plt.grid(True)
st.pyplot(plt)

last_100_days = scaled_data[-100:]

last_100_days = last_100_days.reshape(1, 100, 1)

prediction_10_days = []
for i in range(10):
  next_day_pred = model.predict(last_100_days)
  prediction_10_days.append(next_day_pred)
  last_100_days = np.append(last_100_days[:,1:,:], next_day_pred.reshape(1,1,1), axis=1) # last_100_days[:,1:,:] This selects all data from last_100_days except for the first day.

prediction_10_days = np.array(prediction_10_days)
prediction_10_days = prediction_10_days.reshape(-1, 1)
prediction_10_days = scaler.inverse_transform(prediction_10_days)

last_date = df.index[-1]
next_10_days = pd.date_range(last_date + pd.DateOffset(days=1), periods=10)

plotting_data_10 = pd.DataFrame({
    "Date": next_10_days,
    "Predicted": prediction_10_days.reshape(-1)
})

st.subheader('Closing Price with next 10 days Prediction : ')

plotting_data_all = pd.concat([ploting_data, plotting_data_10])

plotting_data_all['Date'] = pd.to_datetime(plotting_data_all['Date'])


plt.figure(figsize=(15, 5))


plt.plot(ploting_data['Date'], ploting_data['Actual'], label='Actual')


plt.plot(plotting_data_all['Date'], plotting_data_all['Predicted'], label='Predicted')

plt.plot(plotting_data_10['Date'], plotting_data_10['Predicted'], label='10 Days Prediction', color='red')


plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.title("Closing Price with 10 Days Prediction")
plt.legend()
plt.tight_layout()
plt.grid(True)
st.pyplot(plt)

st.subheader("zoomed graph of next 10 days prediction alone : ")
plt.figure(figsize=(15, 5))
plt.plot(plotting_data_10['Date'], plotting_data_10['Predicted'], label='Predicted',color='red')
plt.xlabel("Date")
plt.ylabel("Closing Price Predicted")
plt.title("Next 10 Days Prediction")
plt.legend()
plt.tight_layout()
plt.grid(True)
st.pyplot(plt)

st.subheader("Caution! ")
st.write("Please Don't use the above predictions for real time trading , this are only for educational purposes. Thank You ! ")