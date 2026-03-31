import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv("final_perfect_dataset.csv")

features = [
    'room_temperature',
    'room_humidity',
    'lighting_intensity',
    'room_air_quality',
    'room_CO2'
]

X = df[features]
y = df['comfort_label']

# ---------------------------
# ENCODE LABELS
# ---------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ---------------------------
# SPLIT DATA
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ---------------------------
# MODEL (BEST FOR YOUR CASE)
# ---------------------------
model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

# 🔥 NO SCALING NEEDED for RandomForest
model.fit(X_train, y_train)

# ---------------------------
# EVALUATE
# ---------------------------
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"\n✅ Accuracy: {acc:.4f}")

# ---------------------------
# SAVE MODEL (FINAL FORMAT)
# ---------------------------
with open("model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "label_encoder": le,
        "scaler": None   # IMPORTANT
    }, f)

print("✅ Model saved as model.pkl")