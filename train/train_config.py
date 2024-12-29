from datetime import datetime, timezone, time, timedelta
from dateutil.relativedelta import relativedelta

class TrainingData:
    def __init__(self):
        pass
    ticker = []
    historical_data = []


class Config:
    """Training configuration class."""
    # ---- Modifiable ------
    CONTINUE_LIVE_MODE_AFTER_TRAINING = True
    TRAINING_TIMESTAMP_START_OFFSET = timedelta(days=30)
    STOCK_EXCHANGE_OPENING_TIMESTAMP = time(hour=14, minute=30, second=0) # 9:30 in GMT-5
    STOCK_EXCHANGE_CLOSING_TIMESTAMP = time(hour=21, minute=0, second=0) # 16:00 in GMT-5
    TRAINING_DATA_START = relativedelta(years=1)
    # ---- Will be filled automatically ------
    TRAINING = False
    TRAINING_DATA = TrainingData()
    CURRENT_TRAINING_TIMESTAMP = datetime.now(timezone.utc)
    TRAINING_END_TIMESTAMP = datetime(year=1970, day=1, month=1, tzinfo=timezone.utc)

    @staticmethod
    def is_stock_exchange_open(date):
        return Config.STOCK_EXCHANGE_OPENING_TIMESTAMP <= date.time() <= Config.STOCK_EXCHANGE_CLOSING_TIMESTAMP
