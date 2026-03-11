import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# 1. Load dataset
df = pd.read_csv("modified_indoor_dataset (1).csv")

# 2. Define features (NO occupancy / num_people)
features = [
    'room_temperature',
    'room_humidity',
    'lighting_intensity',
    'room_air_quality',
    'room_CO2'
]

X = df[features]
y = df['comfort_label']

# 3. Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# 5. Train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# 6. Accuracy report
print("--- Model Accuracy Report ---")
print(classification_report(y_test, model.predict(X_test)))

# 7. Save model
with open("model.pkl", "wb") as f:
    pickle.dump((model, le), f)

print("\nSUCCESS: model.pkl trained without occupancy data.")