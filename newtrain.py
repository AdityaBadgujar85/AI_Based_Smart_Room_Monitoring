import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Optional (best performer usually)
from xgboost import XGBClassifier

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv("india_smart_room_dataset_v2.csv")

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
# SCALING (for some models)
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# MODELS TO COMPARE
# ---------------------------
models = {

    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    ),

    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        eval_metric="logloss",
        random_state=42
    ),

    "LogisticRegression": LogisticRegression(
        max_iter=1000
    ),

    "SVM": SVC(
        kernel='rbf',
        probability=True
    )
}

# ---------------------------
# TRAIN + COMPARE
# ---------------------------
results = {}

for name, model in models.items():

    # Use scaled data for LR + SVM
    if name in ["LogisticRegression", "SVM"]:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    results[name] = acc

    print(f"{name} Accuracy: {acc:.4f}")

# ---------------------------
# BEST MODEL SELECTION
# ---------------------------
best_model_name = max(results, key=results.get)
best_accuracy = results[best_model_name]

print("\n🏆 Best Model:", best_model_name)
print("🔥 Best Accuracy:", round(best_accuracy, 4))

best_model = models[best_model_name]

# ---------------------------
# SAVE BEST MODEL
# ---------------------------
with open("best_model.pkl", "wb") as f:
    pickle.dump({
        "model": best_model,
        "label_encoder": le,
        "scaler": scaler if best_model_name in ["LogisticRegression", "SVM"] else None,
        "model_name": best_model_name
    }, f)

print("✅ Best model saved as best_model.pkl")