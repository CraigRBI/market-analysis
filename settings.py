SOURCE_FILE = "YAHOO-INDEX_GSPC.csv"
DATA_DIR = "data"
PROCESSED_DIR = "processed"
TARGET = "Movement"
NON_PREDICTORS = ["Date", "Open", "Close", "High", "Low", "Movement", "Volume", "Adjusted Close"]
BACK_TEST = True
BT_TRAIN_SIZE = 0.7
