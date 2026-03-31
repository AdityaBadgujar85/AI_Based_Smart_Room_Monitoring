import pandas as pd
import random

data = []

comfortable_target = 2000
uncomfortable_target = 2000

comfortable_count = 0
uncomfortable_count = 0

while comfortable_count < comfortable_target or uncomfortable_count < uncomfortable_target:

    # ---------------------------
    # Generate realistic values
    # ---------------------------
    temp = round(random.uniform(18, 38), 1)
    humidity = round(random.uniform(25, 75), 1)
    light = round(random.uniform(100, 800), 1)
    aqi = round(random.uniform(0, 150), 1)
    co2 = round(random.uniform(350, 1400), 1)

    # ---------------------------
    # STRICT REAL-WORLD LOGIC
    # ---------------------------
    comfortable = (
        20 <= temp <= 26 and
        30 <= humidity <= 60 and
        150 <= light <= 500 and
        aqi <= 50 and
        co2 <= 800
    )

    # ---------------------------
    # Label assignment (NO confusion)
    # ---------------------------
    if comfortable and comfortable_count < comfortable_target:
        label = "Comfortable"
        comfortable_count += 1

    elif not comfortable and uncomfortable_count < uncomfortable_target:
        label = "Uncomfortable"
        uncomfortable_count += 1

    else:
        continue

    data.append([temp, humidity, light, aqi, co2, label])


# ---------------------------
# Create DataFrame
# ---------------------------
df = pd.DataFrame(data, columns=[
    'room_temperature',
    'room_humidity',
    'lighting_intensity',
    'room_air_quality',
    'room_CO2',
    'comfort_label'
])

# ---------------------------
# Save dataset
# ---------------------------
df.to_csv("final_perfect_dataset.csv", index=False)

# ---------------------------
# Verification
# ---------------------------
print("\nDataset Distribution:")
print(df['comfort_label'].value_counts())

print("\n✅ Perfect dataset created successfully!")