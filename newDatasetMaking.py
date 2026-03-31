import pandas as pd
import random

data = []

comfortable_target = 2000
uncomfortable_target = 2000

comfortable_count = 0
uncomfortable_count = 0

while comfortable_count < comfortable_target or uncomfortable_count < uncomfortable_target:

    # ---------------------------
    # 🇮🇳 REALISTIC INDIAN DISTRIBUTION
    # ---------------------------

    # 🌡 Temperature (AC + non-AC mix)
    temp = round(random.gauss(28, 3), 1)
    temp = max(20, min(temp, 38))

    # 💧 Humidity (India high + monsoon effect)
    humidity = round(random.gauss(65, 18), 1)
    humidity = max(35, min(humidity, 95))

    # 💡 Light (real indoor rooms, not office lighting)
    light = round(random.gauss(250, 150), 1)
    light = max(50, min(light, 800))

    # 🌫 AQI (India realistic indoor/outdoor mix)
    aqi = round(random.gauss(95, 25), 1)
    aqi = max(40, min(aqi, 180))

    # 🫁 CO2 (normal room vs crowded)
    co2 = round(random.gauss(550, 200), 1)
    co2 = max(350, min(co2, 1800))

    # ---------------------------
    # 🇮🇳 IMPROVED COMFORT LOGIC
    # ---------------------------
    comfortable = (
        24 <= temp <= 30 and
        40 <= humidity <= 75 and
        100 <= light <= 500 and
        aqi <= 110 and
        co2 <= 900
    )

    # ---------------------------
    # LABEL ASSIGNMENT
    # ---------------------------
    if comfortable and comfortable_count < comfortable_target:
        label = "Comfortable"
        comfortable_count += 1

    elif not comfortable and uncomfortable_count < uncomfortable_target:
        label = "Uncomfortable"
        uncomfortable_count += 1

    else:
        continue

    # ---------------------------
    # SENSOR NOISE (realistic)
    # ---------------------------
    temp += random.uniform(-0.5, 0.5)
    humidity += random.uniform(-2, 2)
    light += random.uniform(-20, 20)
    aqi += random.uniform(-5, 5)
    co2 += random.uniform(-30, 30)

    data.append([
        round(temp, 1),
        round(humidity, 1),
        round(light, 1),
        round(aqi, 1),
        round(co2, 1),
        label
    ])

# ---------------------------
# CREATE DATAFRAME
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
# SAVE DATASET
# ---------------------------
df.to_csv("india_smart_room_dataset_v2.csv", index=False)

# ---------------------------
# VERIFICATION
# ---------------------------
print("\nDataset Distribution:")
print(df['comfort_label'].value_counts())

print("\n✅ Realistic Indian Indoor Dataset Created!")