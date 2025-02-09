import datetime

from pymongo import MongoClient

from train.train_config import Config


class TradingSignalLogger:
    def __init__(self, mongo_url):
        """
        Initializes the TradingSignalLogger.
        Args:

        """
        self.client = MongoClient(mongo_url)
        self.db = self.client.signals
        self.collection = self.db.signals_history
        self.strategies = {}

    def log_signal(self, strategy_name, ticker, action, details=None, only_change=False):
        """
        Logs a trading signal into the MongoDB collection.
        Args:
            strategy_name (str): Name of the strategy generating the signal.
            action (str): Type of signal ('buy' or 'sell').
            details (dict): Additional metadata (e.g., price, volume).
            only_change (bool): only log signal changes
        """
        if only_change:
            strategy_dict = self.strategies.get(strategy_name, {})
            last_action = strategy_dict.get(ticker, "")
            if last_action != action:
                strategy_dict[ticker] = action
                self.strategies[strategy_name] = strategy_dict
            else:
                return
        # Document structure
        signal = {
            "strategy": strategy_name,
            "ticker": ticker,
            "action": action,
            "timestamp": Config.CURRENT_TRAINING_TIMESTAMP if Config.TRAINING else datetime.datetime.utcnow(),
            "details": details or {}
        }

        # Insert the signal into the collection
        self.collection.insert_one(signal)

    def get_signals_by_strategy(self, strategy_name):
        """
        Retrieves all signals for a given strategy.
        Args:
            strategy_name (str): Name of the strategy.
        Returns:
            list: List of signals.
        """
        return list(self.collection.find({"strategy": strategy_name}))

    def get_signals_by_ticker(self, ticker):
        """
        Retrieves all signals for a given ticker.
        Args:
            ticker (str): Name of the ticker.
        Returns:
            list: List of signals.
        """
        return list(self.collection.find({"ticker": ticker}))

    def get_signals_by_type(self, signal_type):
        """
        Retrieves all signals of a specific type (buy/sell).
        Args:
            signal_type (str): Type of signal ('buy' or 'sell').
        Returns:
            list: List of signals.
        """
        return list(self.collection.find({"action": signal_type}))

    def delete_signals(self, strategy_name=None):
        """
        Deletes signals for a specific strategy or all signals.
        Args:
            strategy_name (str, optional): Name of the strategy. If None, deletes all signals.
        """
        query = {"strategy": strategy_name} if strategy_name else {}
        self.collection.delete_many(query)