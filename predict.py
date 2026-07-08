import os
import joblib

from src.preprocess import clean_text


class TicketPredictor:

    def __init__(self):

        base_dir = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )

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

    def predict(self, subject, description):

        text = subject + " " + description

        cleaned = clean_text(text)

        vector = self.tfidf.transform([cleaned])

        category = self.category_model.predict(vector)[0]

        priority = self.priority_model.predict(vector)[0]

        return category, priority


def demo():

    predictor = TicketPredictor()

    print("=" * 60)
    print("SUPPORT TICKET PREDICTION")
    print("=" * 60)

    while True:

        subject = input("\nEnter Ticket Subject : ")

        description = input("Enter Ticket Description : ")

        category, priority = predictor.predict(
            subject,
            description
        )

        print("\nPrediction Result")
        print("-" * 30)
        print("Category :", category)
        print("Priority :", priority)

        choice = input("\nPredict another ticket? (y/n): ")

        if choice.lower() != "y":
            print("\nThank you for using the system!")
            break

if __name__ == "__main__":
    demo()