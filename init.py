import os
import joblib

from src.preprocess import clean_text


class TicketPredictor:

    def __init__(self):

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        models_dir = os.path.join(base_dir, "models")

        self.category_model = joblib.load(
            os.path.join(models_dir, "category_model.pkl")
        )

        self.priority_model = joblib.load(
            os.path.join(models_dir, "priority_model.pkl")
        )

        self.tfidf = joblib.load(
            os.path.join(models_dir, "tfidf.pkl")
        )