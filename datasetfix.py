import pandas as pd

# Load your dataset
df = pd.read_csv("perfect_indoor_dataset.csv")

def comfort_rule(row):
    if (
        22 <= row["room_temperature"] <= 26 and
        40 <= row["room_humidity"] <= 60 and
        300 <= row["lighting_intensity"] <= 700 and
        row["room_air_quality"] <= 100 and
        row["room_CO2"] <= 800
    ):
        return "Comfortable"
    else:
        return "Uncomfortable"

# Recalculate label
df["comfort_label"] = df.apply(comfort_rule, axis=1)

# Save new dataset
df.to_csv("perfect_indoor_dataset_clean.csv", index=False)

print("Clean dataset created successfully!")