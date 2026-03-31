from flask import Flask, render_template, jsonify
import pickle
import pandas as pd
import serial
import time
from llm import generate_ai_insight

app = Flask(__name__)

# ---------------------------
# 🔹 GLOBAL VARIABLES
# ---------------------------
ser = None
last_data = None   # 🔥 FIXED (important)


# ---------------------------
# 🔹 INIT SERIAL (SAFE + RETRY)
# ---------------------------
def init_serial():
    global ser

    while True:
        try:
            ser = serial.Serial('COM10', 9600, timeout=1)
            time.sleep(2)  # allow Arduino reset

            # ✅ NOW safe to flush
            ser.reset_input_buffer()
            ser.reset_output_buffer()

            print("✅ Serial connected")
            break

        except Exception as e:
            print("⏳ Waiting for COM port...", e)
            time.sleep(2)


# ---------------------------
# 🔹 LOAD MODEL
# ---------------------------
try:
    with open("best_model.pkl", "rb") as f:
        saved = pickle.load(f)

    model = saved["model"]
    le = saved["label_encoder"]
    scaler = saved["scaler"]

    print("✅ Model loaded successfully")

    test_sample = pd.DataFrame([{
        "room_temperature": 22,
        "room_humidity": 45,
        "lighting_intensity": 400,
        "room_air_quality": 30,
        "room_CO2": 500
    }])

    test_pred = le.inverse_transform(model.predict(test_sample))[0]

    print("🔥 TEST INPUT (22,45,400,30,500)")
    print("🔥 TEST PREDICTION:", test_pred)

except Exception as e:
    print("❌ Model loading error:", e)
    model, le, scaler = None, None, None


# ---------------------------
# 🔹 READ SERIAL DATA (STABLE)
def get_sensor_data():
    global ser, last_data

    if not ser or not ser.is_open:
        return last_data

    try:
        line = ser.readline().decode(errors='ignore').strip()

        # 🔥 If empty → try to recover
        if not line:
            return last_data

        if line.startswith("DATA:"):
            print("RAW:", line)

            line = line.replace("DATA:", "")
            values = line.split(",")

            if len(values) == 5:
                data = {
                    "temperature": float(values[0]),
                    "humidity": float(values[1]),
                    "light": float(values[2]),
                    "air": float(values[3]),
                    "co2": float(values[4])
                }

                print("✅ NEW DATA:", data)

                last_data = data
                return data

    except Exception as e:
        print("❌ Serial error:", e)

        # 🔥 HARD RESET if serial stuck
        try:
            ser.reset_input_buffer()
        except:
            pass

    # 🔥 fallback
    if last_data:
        print("⚠️ Using last valid data")
        return last_data

    return None


# ---------------------------
# 🔹 MAIN DASHBOARD ROUTE
# ---------------------------
@app.route("/")
def dashboard():

    data = get_sensor_data()

    if not data:
        data = {
            "temperature": 0,
            "humidity": 0,
            "light": 0,
            "air": 0,
            "co2": 0
        }

    print("\n📥 Live Sensor Data:", data)

    prediction = None
    confidence = None
    ai_response = None

    sample = pd.DataFrame([{
        "room_temperature": data["temperature"],
        "room_humidity": data["humidity"],
        "lighting_intensity": data["light"],
        "room_air_quality": data["air"],
        "room_CO2": data["co2"]
    }])

    if model and le:
        result = model.predict(sample)
        prediction = le.inverse_transform(result)[0]

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(sample)
            confidence = round(max(prob[0]) * 100, 2)

        print("🔮 Prediction:", prediction)
        print("📊 Confidence:", confidence)

    else:
        prediction = "Model not loaded"

    ai_response = generate_ai_insight(
        data=data,
        ml_prediction=prediction
    )

    return render_template(
        "dashboard.html",
        data=data,
        prediction=prediction,
        confidence=confidence,
        ai_response=ai_response
    )


# ---------------------------
# 🔹 API ROUTE (REAL-TIME)
# ---------------------------
@app.route("/api/data")
def api_data():

    data = get_sensor_data()

    if not data:
        data = {
            "temperature": 0,
            "humidity": 0,
            "light": 0,
            "air": 0,
            "co2": 0
        }

    prediction = None

    sample = pd.DataFrame([{
        "room_temperature": data["temperature"],
        "room_humidity": data["humidity"],
        "lighting_intensity": data["light"],
        "room_air_quality": data["air"],
        "room_CO2": data["co2"]
    }])

    if model and le:
        result = model.predict(sample)
        prediction = le.inverse_transform(result)[0]

    return jsonify({
        "data": data,
        "prediction": prediction
    })


# ---------------------------
# 🔹 RUN APP
# ---------------------------
if __name__ == "__main__":
    init_serial()
    app.run(debug=True, use_reloader=False)