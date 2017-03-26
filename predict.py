import os
import settings
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


def read(source):
    data = pd.read_csv(os.path.join(settings.PROCESSED_DIR, source), parse_dates=['Date'])
    return data


def predict(train, test):
    target = settings.TARGET
    predictors = [p for p in train.columns.tolist() if p not in settings.NON_PREDICTORS]
    model = None

    if target is "Movement":
        model = LogisticRegression()
    elif target is "Close":
        model = LinearRegression()
    else:
        return

    # Train
    model.fit(train[predictors], train[target])

    # Predict
    prediction = model.predict(test[predictors])

    return prediction


# Currently only have model validation for logistic regression model;
# forward chaining algorithm returns model accuracy for predicting next
# price movement in time series data.
def back_test(train, test):
    target = "Movement"
    predictors = [p for p in train.columns.tolist() if p not in settings.NON_PREDICTORS]

    predicted_movement = []
    true_movement = []

    updated_train = train
    for idx, updated_test in test.iterrows():
        model = LogisticRegression()

        # Train
        model.fit(updated_train[predictors], updated_train[target])

        # Predict
        predicted_movement.append(model.predict([updated_test[predictors]]))

        true_movement.append(updated_test["Movement"])
        updated_train = updated_train.append([updated_test])

    correct_predictions = pd.Series(true_movement) == pd.Series(predicted_movement)
    accuracy = pd.Series(predicted_movement)[correct_predictions].shape[0] / pd.Series(predicted_movement).shape[0]
    test["Predicted Movement"] = predicted_movement
    test.to_csv(os.path.join(settings.PROCESSED_DIR, "predictions.csv"), index=False)
    return accuracy


if __name__ == "__main__":
    if not settings.BACK_TEST:
        train = read("train.csv")
        test = read("test.csv")
        result = predict(train, test)

        print("Results -")
        print(result)
    else:
        train = read("train_bt.csv")
        test = read("test_bt.csv")
        accuracy = back_test(train, test)

        print("Results -")
        print("Model accuracy: {}".format(accuracy))