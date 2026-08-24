import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

CSV_PATH = "data/landmarks.csv"
MODEL_PATH = "model/sign_model.pkl"
ENCODER_PATH = "model/label_encoder.pkl"
CLASSES = ["Hello", "Yes", "No", "Help", "Thank You", "Please", "Sorry", "Welcome", "Stop", "Good", "Bad", "Wait"]

os.makedirs("model", exist_ok=True)


def train():
    print("SignBridge AI - Model Training")
    print("================================")

    # 1. Load dataset
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: Dataset not found at {CSV_PATH}")
        print("Please run: python collect_data.py")
        sys.exit(1)

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"ERROR: Failed to read CSV: {e}")
        sys.exit(1)

    if df.empty:
        print("ERROR: Dataset is empty.")
        sys.exit(1)

    print(f"Total samples loaded: {len(df)}")
    print(f"Columns: {list(df.columns[:5])} ... ({len(df.columns)} total)")

    # 2. Validate classes
    found_classes = sorted(df["label"].unique().tolist())
    print(f"Classes found: {found_classes}")

    missing = [c for c in CLASSES if c not in found_classes]
    if missing:
        print(f"WARNING: Missing classes: {missing}")
        print("Continuing with available classes...")

    if len(found_classes) < 2:
        print("ERROR: Need at least 2 classes to train.")
        sys.exit(1)

    # 3. Per-class counts
    print("\nSamples per class:")
    for cls in found_classes:
        count = len(df[df["label"] == cls])
        print(f"  {cls}: {count}")

    # 4. Separate features and labels
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32)
    y_raw = df["label"].values

    # Drop rows with NaN
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    y_raw = y_raw[valid_mask]
    print(f"\nValid samples after NaN drop: {len(X)}")

    # 5. Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"Encoded classes: {list(le.classes_)}")

    # 6. Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    # 7. Train RandomForestClassifier
    print("\nTraining RandomForestClassifier...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # 8. Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 9. Save model and encoder
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Encoder saved to: {ENCODER_PATH}")

    # 10. Summary
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print(f"  Classes: {list(le.classes_)}")
    print(f"  Total samples: {len(X)}")
    print(f"  Training: {len(X_train)}, Testing: {len(X_test)}")
    print(f"  Accuracy: {acc * 100:.2f}%")
    print(f"  Model: {MODEL_PATH}")
    print("="*50)

    return acc


if __name__ == "__main__":
    train()