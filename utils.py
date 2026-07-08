import os
import joblib


def create_directories():

    folders = [

        "models",

        "outputs"

    ]

    for folder in folders:

        os.makedirs(folder, exist_ok=True)


def save_model(model, filename):

    joblib.dump(model, filename)

    print(f"\nModel saved successfully -> {filename}")


def load_model(filename):

    return joblib.load(filename)