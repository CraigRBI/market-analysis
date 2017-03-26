import os
import settings
import pandas as pd
from sklearn.linear_model import LinearRegression


def read(source):
    data = pd.read_csv(os.path.join(settings.PROCESSED_DIR, source), parse_dates=['Date'])
    return data


def predict(train, test):
    target = settings.TARGET
    clf = LinearRegression()

    predictors = [p for p in train.columns.tolist() if p not in settings.NON_PREDICTORS]
    clf.fit(train[predictors], train[target])

    predicted_close = clf.predict(test[predictors])
    predicted_movement = None
    if predicted_close >= train.iloc[-1]["Close"]:
        predicted_movement = "Bullish"
    else:
        predicted_movement = "Bearish"

    return [predicted_close, predicted_movement]


def back_test(train, test):
    # 'Forward chaining' algorithm returns model accuracy for predicting next price movement in time series data
    target = "Close"
    predictors = [p for p in train.columns.tolist() if p not in settings.NON_PREDICTORS]

    predicted_movement = []
    true_movement = []

    predicted_close = []
    true_close = []

    updated_train = train
    prev_test = train.iloc[-1]
    for idx, updated_test in test.iterrows():
        clf = LinearRegression()

        # Train
        clf.fit(updated_train[predictors], updated_train[target])

        # Predict
        close_prediction = clf.predict([updated_test[predictors]])
        predicted_close.append(close_prediction)
        if close_prediction >= prev_test["Close"]:
            predicted_movement.append(1)                            # Bullish movement
        else:
            predicted_movement.append(0)                            # Bearish movement

        true_movement.append(updated_test["Movement"])
        true_close.append(updated_test["Close"])

        updated_train = updated_train.append([updated_test])
        prev_test = updated_test

    correct_predictions = pd.Series(true_movement) == pd.Series(predicted_movement)
    accuracy = pd.Series(predicted_movement)[correct_predictions].shape[0] / pd.Series(predicted_movement).shape[0]
    test["Predicted Movement"] = predicted_movement
    test["Predicted Close"] = predicted_close
    test.to_csv(os.path.join(settings.PROCESSED_DIR, "predictions.csv"), index=False)
    return accuracy


if __name__ == "__main__":
    if not settings.BACK_TEST:
        train = read("train.csv")
        test = read("test.csv")
        results = predict(train, test)

        print("Results -")
        print("Close price: {}".format(results[0]))
        print("Price movement: {}".format(results[1]))
    else:
        train = read("train_bt.csv")
        test = read("test_bt.csv")
        accuracy = back_test(train, test)

        print("Results -")
        print("Model accuracy: {}".format(accuracy))