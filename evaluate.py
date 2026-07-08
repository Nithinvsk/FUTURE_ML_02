import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)


def evaluate_model(
        model,
        X_test,
        y_test,
        class_names,
        title,
        output_prefix
):
    """
    Evaluate a trained classification model.
    Saves:
        - Confusion Matrix (.png)
        - Classification Report (.txt)
    """

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")

    report = classification_report(
        y_test,
        predictions
    )

    print(report)

    os.makedirs("outputs", exist_ok=True)

    report_path = f"outputs/{output_prefix}_report.txt"

    with open(report_path, "w") as file:

        file.write(f"{title}\n\n")

        file.write(f"Accuracy : {accuracy:.4f}\n\n")

        file.write(report)

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title(title)

    image_path = f"outputs/{output_prefix}_confusion_matrix.png"

    plt.savefig(image_path)

    plt.close()

    print(f"\nSaved : {report_path}")

    print(f"Saved : {image_path}")