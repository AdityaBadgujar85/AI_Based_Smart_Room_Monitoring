from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
try:
    with open("model.pkl", "rb") as f:
        model, le = pickle.load(f)
except Exception as e:
    print("Model loading error:", e)
    model, le = None, None


@app.route("/", methods=["GET", "POST"])
def dashboard():

    data = None
    prediction = None

    if request.method == "POST":
        try:
            # Get form values
            data = {
                "temperature": float(request.form.get("temperature")),
                "humidity": float(request.form.get("humidity")),
                "light": float(request.form.get("light")),
                "air": float(request.form.get("air")),
                "co2": float(request.form.get("co2"))
            }

            # Prepare dataframe for model
            sample = pd.DataFrame([{
                "room_temperature": data["temperature"],
                "room_humidity": data["humidity"],
                "lighting_intensity": data["light"],
                "room_air_quality": data["air"],
                "room_CO2": data["co2"]
            }])

            # Predict comfort
            if model is not None and le is not None:
                result = model.predict(sample)
                prediction = le.inverse_transform(result)[0]

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("dashboard.html", data=data, prediction=prediction)


if __name__ == "__main__":
    # IMPORTANT for Docker
    app.run(host="0.0.0.0", port=5000, debug=True)