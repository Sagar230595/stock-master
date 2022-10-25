import streamlit as st
import pandas as pd
import numpy as np
import datetime
import nsepy as nse
import pandas_ta as ta
from dateutil.relativedelta import relativedelta
import nsepython as nsep
import pickle
import sklearn.metrics as metrics

Min_max_scalar = pickle.load(open('MinMaxscalar.pkl', 'rb'))
sgd_reg_model = pickle.load(open('sgdregressor.pkl', 'rb'))

st.title('Nifty50 Prediction System')

# https://docs.streamlit.io/library/api-reference/widgets/st.date_input
input_date = st.date_input(
    "Enter date",
    datetime.date(2022, 12, 6))


def prediction(tomorrow_date):
    # https://www.geeksforgeeks.org/how-to-add-and-subtract-days-using-datetime-in-python/
    today_date = tomorrow_date - relativedelta(days=1)
    ohlc_df = nse.get_history(symbol="NIFTY", start=datetime.date(2018, 1, 1), end=today_date, index=True)
    ohlc_df = ohlc_df.reset_index()
    ohlc_df = ohlc_df.drop(['Volume','Turnover'], axis=1)
    # # https://pynative.com/python-datetime-format-strftime/#:~:text=Use%20datetime.,hh%3Amm%3Ass%20format.
    today_date_str = today_date.strftime("%d-%B-%Y")
    return_index = nsep.index_total_returns(symbol="NIFTY 50", start_date=today_date_str, end_date=today_date_str)
    ohlc_df['EMA200'] = ta.ema(ohlc_df['Close'],200)
    ohlc_df['EMA10'] = ta.ema(ohlc_df['Close'], 10)
    ohlc_df['BBlower'] = ta.bbands(ohlc_df['Close'], 20)['BBL_20_2.0'].values
    ohlc_df['BBupper'] = ta.bbands(ohlc_df['Close'], 20)['BBU_20_2.0'].values
    ohlc_df['SMA10'] = ta.sma(ohlc_df['Close'],10)
    final_df_columns = ['Close', 'Total_return_index', 'Low', 'High', 'EMA200', 'SMA10', 'BB_lower', 'Open',
                        'BB_upper', 'EMA10']
    final_df = pd.DataFrame(columns = final_df_columns)
    final_df['Close'] = ohlc_df['Close'].values
    final_df['Low'] = ohlc_df['Low'].values
    final_df['High'] = ohlc_df['High'].values
    final_df['EMA200'] = ohlc_df['EMA200'].values
    final_df['SMA10'] = ohlc_df['SMA10'].values
    final_df['BB_lower'] = ohlc_df['BBlower'].values
    final_df['Open'] = ohlc_df['Open'].values
    final_df['BB_upper'] = ohlc_df['BBupper'].values
    final_df['EMA10'] = ohlc_df['EMA10'].values
    final_df = final_df[-1:]
    final_df['Total_return_index'] = return_index['TotalReturnsIndex'].values
    d = Min_max_scalar.transform(final_df)
    final_df_scaled = pd.DataFrame(d, columns=final_df_columns)
    pred_price = sgd_reg_model.predict(final_df_scaled)
    return pred_price


st.text('Predicted Price of Nifty Index for given date is')
if st.button('Predict'):
    predicted_price = prediction(input_date)
    st.write(predicted_price)


def error_in_pred():
    actual_value = nse.get_history(symbol="NIFTY", start=input_date, end=input_date, index=True)['Close'].values
    y_pred = prediction(input_date)
    if len(actual_value) == 0:
        return 'Closing price is not available. You need to wait for the day to end'
    else:
        return metrics.mean_squared_error(actual_value, y_pred, squared=False)


st.text('Error in prediction of Nifty Price is')
if st.button('Error'):
    calculated_error = error_in_pred()
    st.write(calculated_error)