import re
import nltk
import pandas as pd

from nltk.corpus import stopwords

# Download resources only if they don't already exist
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))


def clean_text(text):
    """
    Clean support ticket text.

    Steps:
    1. Convert to lowercase
    2. Remove punctuation
    3. Remove numbers
    4. Remove stopwords
    """

    text = str(text).lower()

    text = re.sub(r"[^a-zA-Z ]", " ", text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)


def load_dataset(path):

    df = pd.read_csv(path)

    print("=" * 60)
    print("DATASET LOADED SUCCESSFULLY")
    print("=" * 60)

    print(df.head())

    print("\nColumns:\n")

    print(df.columns)

    return df


def prepare_text(df):

    df["combined_text"] = (

            df["Ticket Subject"].fillna("")
            + " "
            + df["Ticket Description"].fillna("")

    )

    df["clean_text"] = df["combined_text"].apply(clean_text)

    return df