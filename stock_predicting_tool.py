from sklearn.tree import DecisionTreeClassifier
import joblib
import yfinance as yf
import pandas as pd
import numpy as np

def main():
    # loadinf model
    model = joblib.load('stock_prediction_tool.joblib')

    # querying the information
    ticker = yf.Ticker('NVDA')

    #taking the information under the y finance format, putting max period to have the data to do our calculations
    unformatted_data = ticker.history(period='max')

    #foramtting the data into a list understandable to the model
    data = get_financial_indicators(unformatted_data)

    #taking the last row of this list of rows which should be todats row
    todays_row = data.tail(1)

    #extracting the values of the last row
    predicting_list = todays_row.values

    #testing
    print(todays_row)
    print(model.predict(predicting_list))


def get_financial_indicators(data):
    # Volume
    data['Volume'] = data['Volume']

    # Moving Average 10
    data['MA_10'] = data['Close'].rolling(window=10).mean()

    # Relative Strength Index (RSI) 14
    delta = data['Close'].diff()
    gain = delta.where(delta >= 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI_14'] = 100 - (100 / (1 + rs))

    # Moving Average 20
    data['MA_20'] = data['Close'].rolling(window=20).mean()

    # Bollinger Bands
    std_dev = data['Close'].rolling(window=20).std()
    data['Upper_Band'] = data['MA_20'] + 2 * std_dev
    data['Lower_Band'] = data['MA_20'] - 2 * std_dev

    # Exponential Moving Average (EMA) 12 and 26
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()

    # Moving Average Convergence Divergence (MACD) and Signal Line
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # On Balance Volume (OBV)
    data['OBV'] = np.where(data['Close'].diff() > 0, data['Volume'], 
                           np.where(data['Close'].diff() < 0, -data['Volume'], 0)).cumsum()

    # True Range
    data['TR1'] = abs(data['High'] - data['Low'])
    data['TR2'] = abs(data['High'] - data['Close'].shift())
    data['TR3'] = abs(data['Low'] - data['Close'].shift())
    data['True_Range'] = data[['TR1', 'TR2', 'TR3']].max(axis=1)

    # Average True Range (ATR) 14
    data['ATR_14'] = data['True_Range'].rolling(window=14).mean()

    #adding previous data
    data['Prev_Open'] = data['Open'].shift()
    data['Prev_High'] = data['High'].shift()
    data['Prev_Low'] = data['Low'].shift()
    data['Prev_Close'] = data['Close'].shift()


    return data.iloc[:, 4:].drop(columns=['Dividends', 'Stock Splits'])


if __name__ == '__main__':
    main()