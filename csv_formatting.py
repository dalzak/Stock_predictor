import pandas as pd
import numpy as np

# read in the data
df = pd.read_csv("your_file.csv")

# Moving Average (10-day)
df['MA_10'] = df['Close'].rolling(window=10).mean()

# Relative Strength Index (RSI, 14-day)
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI_14'] = 100 - (100 / (1 + rs))

# Bollinger Bands (20-day, 2 standard deviations)
df['MA_20'] = df['Close'].rolling(window=20).mean()
df['Upper_Band'] = df['MA_20'] + 2 * df['Close'].rolling(window=20).std()
df['Lower_Band'] = df['MA_20'] - 2 * df['Close'].rolling(window=20).std()

# Moving Average Convergence Divergence (MACD)
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

# On-Balance Volume (OBV)
df['OBV'] = np.where(df['Close'] > df['Close'].shift(1), df['Volume'], 
                      np.where(df['Close'] < df['Close'].shift(1), -df['Volume'], 0)).cumsum()

# Average True Range (ATR, 14-day)
df['TR1'] = abs(df['High'] - df['Low'])
df['TR2'] = abs(df['High'] - df['Close'].shift())
df['TR3'] = abs(df['Low'] - df['Close'].shift())
df['True_Range'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
df['ATR_14'] = df['True_Range'].rolling(window=14).mean()

#save the updated dataframe
df.to_csv('your_new_df_name.csv', index=False)


