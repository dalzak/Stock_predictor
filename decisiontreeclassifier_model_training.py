import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


def main():
    # load the dataset
    df = pd.read_csv('nvidia_updated.csv')

    # create new columns with the previous day's open, high, low, and close prices
    df['Prev_Open'] = df['Open'].shift(1)
    df['Prev_High'] = df['High'].shift(1)
    df['Prev_Low'] = df['Low'].shift(1)
    df['Prev_Close'] = df['Close'].shift(1)

    # drop the unnecessary columns (high, low, date, adj close)
    df = df.drop(['High', 'Low', 'Date', 'Adj Close'], axis=1)

    # calculate the difference between the close and open prices
    df['Close_Open_Diff'] = df['Close'] - df['Open']

    # apply the sign_label function to each element in the Close_Open_Diff column
    df['Close_Open_Label'] = df['Close_Open_Diff'].apply(sign_label)

    # drop any rows with missing data
    df.dropna(inplace=True)

    # drop the open, close, and close-open difference columns
    df = df.drop(['Open', 'Close', 'Close_Open_Diff'], axis=1)

    # separate the data into input (X) and output (y)
    X = df.drop('Close_Open_Label', axis=1)
    y = df['Close_Open_Label']

    #spliting the dataset in 2 for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # train a decision tree classifier model
    Params = {
        "max_depth": 6,
        "min_samples_split": 8,
        "min_samples_leaf": 9,
        "criterion": "entropy",
        "max_features": None,
        "min_impurity_decrease": 0.004249468939060151
    }
    model = RandomForestClassifier(**Params)
    model.fit(X_train, y_train)


    #our prediction based on the data we have to predict stuff
    prediction = model.predict(X_test)

    #testing the algo through having an accuracy score
    score = accuracy_score(y_test, prediction)

    print(score)


def sign_label(x):
    return "positive" if x >= 0 else "negative"

if __name__ == '__main__':
    main()