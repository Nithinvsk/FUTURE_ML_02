import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def train_category_model(X_train, X_test, y_train, y_test):

    models = {

        "Logistic Regression": LogisticRegression(max_iter=1000),

        "Naive Bayes": MultinomialNB(),

        "Linear SVM": LinearSVC(),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

    }

    best_model = None
    best_accuracy = 0
    best_name = ""

    print("\n" + "=" * 60)
    print("CATEGORY MODEL COMPARISON")
    print("=" * 60)

    for name, model in models.items():

        model.fit(X_train, y_train)

        prediction = model.predict(X_test)

        accuracy = accuracy_score(y_test, prediction)

        print(f"\n{name}")
        print(f"Accuracy : {accuracy:.4f}")

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_model = model
            best_name = name

    print("\n" + "=" * 60)
    print("BEST CATEGORY MODEL")
    print("=" * 60)

    print(best_name)
    print("Accuracy :", round(best_accuracy, 4))

    final_prediction = best_model.predict(X_test)

    print("\nClassification Report\n")

    print(classification_report(y_test, final_prediction))

    joblib.dump(best_model, "models/category_model.pkl")

    print("\ncategory_model.pkl saved.")

    return best_model