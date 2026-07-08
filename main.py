from src.evaluate import evaluate_model
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.preprocess import load_dataset
from src.preprocess import prepare_text
from src.utils import create_directories
from src.train_category import train_category_model
from src.train_priority import train_priority_model


def main():

    print("=" * 70)
    print("SUPPORT TICKET CLASSIFICATION & PRIORITY PREDICTION")
    print("=" * 70)

    ###############################################################
    # Create Required Directories
    ###############################################################

    create_directories()

    ###############################################################
    # Load Dataset
    ###############################################################

    df = load_dataset("dataset/customer_support_tickets.csv")

    ###############################################################
    # Prepare Text
    ###############################################################

    df = prepare_text(df)

    ###############################################################
    # Check Missing Values
    ###############################################################

    print("\nMissing Values\n")

    print(
        df[
            [
                "Ticket Description",
                "Ticket Type",
                "Ticket Priority"
            ]
        ].isnull().sum()
    )

    ###############################################################
    # TF-IDF Vectorization
    ###############################################################

    tfidf = TfidfVectorizer(max_features=5000)

    X = tfidf.fit_transform(df["clean_text"])

    print("\nFeature Matrix Shape:", X.shape)

    ###############################################################
    # CATEGORY MODEL
    ###############################################################

    y_category = df["Ticket Type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_category,
        test_size=0.20,
        random_state=42
    )

    category_model = train_category_model(
        X_train,
        X_test,
        y_train,
        y_test
    )
    evaluate_model(
        category_model,
        X_test,
        y_test,
        category_model.classes_,
        "Category Classification",
        "category"
    )
    ###############################################################
    # PRIORITY MODEL
    ###############################################################

    y_priority = df["Ticket Priority"]

    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
        X,
        y_priority,
        test_size=0.20,
        random_state=42
    )

    priority_model = train_priority_model(
        X_train_p,
        X_test_p,
        y_train_p,
        y_test_p
    )
    evaluate_model(
        priority_model,
        X_test_p,
        y_test_p,
        priority_model.classes_,
        "Priority Classification",
        "priority"
    )
    ###############################################################
    # Save TF-IDF Vectorizer
    ###############################################################

    joblib.dump(tfidf, "models/tfidf.pkl")

    print("\nTF-IDF vectorizer saved successfully.")

    ###############################################################
    # Project Summary
    ###############################################################

    print("\n" + "=" * 70)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("Saved Files:")
    print("✔ models/category_model.pkl")
    print("✔ models/priority_model.pkl")
    print("✔ models/tfidf.pkl")


if __name__ == "__main__":
    main()