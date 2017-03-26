import os
import settings
import pandas as pd


def read(source):
    data = pd.read_csv(os.path.join(settings.DATA_DIR, source), parse_dates=['Date'])
    return data


def transform(data):
    data["Volume"] = data["Volume"].apply(lambda x: int(x))

    for column in [
        "Open",
        "High",
        "Low",
        "Close",
        "Adjusted Close"
    ]:
        data[column] = data[column].apply(lambda x: float(x))

    data.sort_values(by=['Date'], ascending=True, inplace=True)

    # Add column representing movement direction of price
    data["Movement"] = data["Close"] > data["Close"].shift(1)
    data["Movement"] = data["Movement"].apply(lambda x: int(x))

    # Add useful indicators
    data.loc[data.shape[0]] = None
    data["5 Day Mean (Closing Price)"] = data["Close"].rolling(window=5).mean().shift(1)
    data["5 Day Std Deviation (Closing Price)"] = data["Close"].rolling(window=5).std().shift(1)
    data = data.iloc[:-2].dropna().append([data.iloc[-1]])
    return data


def write(data):
        train = data.iloc[:int(round(data.shape[0] * settings.BT_TRAIN_SIZE, 0))]
        train.to_csv(os.path.join(settings.PROCESSED_DIR, "train_bt.csv"), index=False)

        test = data.iloc[int(round(data.shape[0] * settings.BT_TRAIN_SIZE, 0)):-2]
        test.to_csv(os.path.join(settings.PROCESSED_DIR, "test_bt.csv"), index=False)

        data.iloc[:-2].to_csv(os.path.join(settings.PROCESSED_DIR, "train.csv"), index=False)
        data.iloc[[-1]].to_csv(os.path.join(settings.PROCESSED_DIR, "test.csv"), index=False)


if __name__ == "__main__":
    data = read(settings.SOURCE_FILE)
    data = transform(data)
    write(data)
