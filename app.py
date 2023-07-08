import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from keras.models import load_model
import streamlit as st

start = '2010-04-01'
end = '2023-03-31'

st.title('Stock Trend Predictor')
user_input =st.test_input('Enter Stock Ticker','TATASTEEL.NS')

df = yf.download('TATASTEEL.NS', start=start, end=end)
df.head()

# decribe data in front of user
st.subheader('Data from 2010-2023')
st.write(df.describe())

# Visualization of Stock 
st.subheader('Closing Price vs Time chart')
fig =plt.figure(fisize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

# Adding Moving Average
st.subheader('Closing Price vs Time chart with 100 & 200 Moving Average')
ma100=df.Close.rolling(100).mean
ma200=df.Close.rolling(200).mean
fig =plt.figure(fisize=(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'b')
plt.plot(df.Close, 'g')
st.pyplot(fig)

data_train =pd.DataFrame(df['Close'][0:int(len(df)*0.80)])
data_test =pd.DataFrame(df['Close'][int(len(df)*0.80): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler(feature_range=(0,1))
data_train_array=scaler.fit_transform(data_train)


# load my model
model= load_model('keras_stock_model.h5')

# added prev 100 rows so thatb it will predict to first test row
prev_100= data_train.tail(100)
final_test_df = prev_100.append(data_test, ignore_index =True)
input_data= scaler.fit_transform(final_test_df)
x_test =[]
y_test =[]
for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])
x_test, y_test =np.array(x_test), np.array(y_test)

y_pred =model.predict(x_test)
# scaled up
scaler = scaler.scale_
scale_factor= 1/scaler[0]
y_pred =y_pred*scale_factor
y_test =y_test *scale_factor

# Final graph
st.subheader('Predicted Price vs Original Price')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label ='Original Price')
plt.plot(y_pred, 'r', label ='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
