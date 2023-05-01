import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import optuna
import joblib


def main():

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=150)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    joblib.dump(study, "study.pkl")

def sign_label(x):

    return "positive" if x >= 0 else "negative"


def objective(trial):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # define the hyperparameter ranges
    params = {
        'max_depth': trial.suggest_categorical('max_depth', list(range(1, 21))),
        'min_samples_split': trial.suggest_categorical('min_samples_split', list(range(2, 11))),
        'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', list(range(1, 11))),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'min_impurity_decrease': trial.suggest_loguniform('min_impurity_decrease', 1e-8, 1e-2)
    }


    model = RandomForestClassifier(**params, )

    _ = model.fit(X_train, y_train)

    return accuracy_score(y_test, model.predict(X_test))


if __name__ == '__main__':
    main()