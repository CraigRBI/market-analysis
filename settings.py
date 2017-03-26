SOURCE_FILE = "YAHOO-INDEX_GSPC.csv"
DATA_DIR = "data"
PROCESSED_DIR = "processed"
TARGET = "Close"
NON_PREDICTORS = ["Date", "Open", "Close", "High", "Low", "Movement", "Volume", "Adjusted Close"]
BACK_TEST = False
BT_TRAIN_SIZE = 0.9
