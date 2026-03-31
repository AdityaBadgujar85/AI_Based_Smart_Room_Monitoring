from flask import Flask, render_template, jsonify
import pickle
import pandas as pd
import os
import time

# ---------------------------
# SAFE SERIAL IMPORT
# ---------------------------
try:
    import serial
except ImportError:
    serial = None

from llm import generate_ai_insight

app = Flask(__name__)

# ---------------------------
# GLOBAL VARIABLES
# ---------------------------
ser = None
model, le, scaler = None, None, None

last_data = {
    "temperature": 0,
    "humidity": 0,
    "light": 0,
    "air": 0,
    "co2": 0
}

# ---------------------------
# CONFIG
# ---------------------------
SERIAL_PORT = os.getenv("SERIAL_PORT", "COM8")
BAUD_RATE = int(os.getenv("BAUD_RATE", 9600))
DOCKER_MODE = os.getenv("DOCKER", "false").lower() == "true"


# ---------------------------
# SERIAL INIT
# ---------------------------
def init_serial():
    global ser

    if serial is None:
        print("⚠️ pyserial not installed")
        return

    if DOCKER_MODE:
        print("🚀 Docker mode → Serial disabled")
        return

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"✅ Serial connected: {SERIAL_PORT}")
    except Exception as e:
        print("⚠️ Serial error:", e)
        ser = None


# ---------------------------
# MODEL LOADER (FINAL FIX)
# ---------------------------
def load_model():
    global model, le, scaler

    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, "best_model.pkl")

        print("\n📁 Loading model from:", model_path)
        print("📁 File exists:", os.path.exists(model_path))

        if not os.path.exists(model_path):
            print("❌ Model file missing!")
            return

        with open(model_path, "rb") as f:
            saved = pickle.load(f)

        print("🔥 Loaded object type:", type(saved))

        # ✅ HANDLE ALL CASES
        if isinstance(saved, dict):
            model = saved.get("model") or saved.get("clf") or saved.get("estimator")
            le = saved.get("label_encoder")
            scaler = saved.get("scaler")

        else:
            model = saved
            le = None
            scaler = None

        # ✅ FINAL CHECK
        if model is None:
            print("❌ Model is STILL None → BAD FILE")
        else:
            print("✅ Model loaded SUCCESSFULLY")
            print("MODEL TYPE:", type(model))

    except Exception as e:
        print("❌ Model loading FAILED:", e)


# ---------------------------
# SENSOR PARSER
# ---------------------------
def parse_sensor_line(line):
    try:
        if not line.startswith("DATA:"):
            return None

        values = line.replace("DATA:", "").split(",")

        if len(values) != 5:
            return None

        return {
            "temperature": float(values[0]),
            "humidity": float(values[1]),
            "light": float(values[2]),
            "air": float(values[3]),
            "co2": float(values[4])
        }
    except:
        return None


def get_sensor_data():
    global ser, last_data

    if ser is None or not ser.is_open:
        return last_data

    try:
        line = ser.readline().decode(errors='ignore').strip()
        parsed = parse_sensor_line(line)

        if parsed:
            last_data = parsed

    except Exception as e:
        print("❌ Serial read error:", e)

    return last_data


# ---------------------------
# PREPARE INPUT
# ---------------------------
def prepare_input(data):
    df = pd.DataFrame([{
        "room_temperature": data["temperature"],
        "room_humidity": data["humidity"],
        "lighting_intensity": data["light"],
        "room_air_quality": data["air"],
        "room_CO2": data["co2"]
    }])

    if scaler is not None:
        try:
            df = scaler.transform(df)
        except Exception as e:
            print("⚠️ Scaler error:", e)

    return df


# ---------------------------
# PREDICTION (FINAL FIX)
# ---------------------------
def get_prediction(data):

    if model is None:
        return "Model not loaded", None

    try:
        sample = prepare_input(data)

        result = model.predict(sample)

        # ✅ HANDLE NO LABEL ENCODER
        if le is not None:
            prediction = le.inverse_transform(result)[0]
        else:
            prediction = str(result[0])

        confidence = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(sample)
            confidence = round(max(prob[0]) * 100, 2)

        return prediction, confidence

    except Exception as e:
        print("❌ Prediction error:", e)
        return "Error", None


# ---------------------------
# ROUTES
# ---------------------------
@app.route("/")
def dashboard():
    data = get_sensor_data()
    prediction, confidence = get_prediction(data)

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


@app.route("/api/data")
def api_data():
    data = get_sensor_data()
    prediction, _ = get_prediction(data)

    ai_response = generate_ai_insight(
        data=data,
        ml_prediction=prediction
    )

    return jsonify({
        "data": data,
        "prediction": prediction,
        "ai_response": ai_response
    })


# ---------------------------
# START
# ---------------------------
if __name__ == "__main__":
    print("🚀 Starting Smart Room System...")

    load_model()   # 🔥 CRITICAL
    init_serial()

    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)