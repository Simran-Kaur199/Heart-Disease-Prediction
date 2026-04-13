import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from pipeline import create_pipeline


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess(df):
    X = df.drop('target', axis=1)
    y = df['target']
    return X, y


def train():
    # Load data
    df = load_data("data/heart_feature_encoded.csv")

    # Split
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create pipeline
    pipeline = create_pipeline()

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {roc:.4f}")

    # Save model
    with open("heart_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    print("Model saved as heart_pipeline.pkl")


if __name__ == "__main__":
    train()