import serial
import os
import time

# ---------------------------
# 🔹 CONFIG
# ---------------------------
SERIAL_PORT = os.getenv("SERIAL_PORT", "COM8")
BAUD_RATE = int(os.getenv("BAUD_RATE", 9600))

ser = None


# ---------------------------
# 🔹 INIT SERIAL (SAFE)
# ---------------------------
def init_serial():
    global ser

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"✅ Connected to {SERIAL_PORT}")

    except Exception as e:
        print(f"⚠️ Serial not available ({SERIAL_PORT}):", e)
        ser = None


# ---------------------------
# 🔹 READ SENSOR DATA
# ---------------------------
def get_sensor_data():
    global ser

    if ser is None or not ser.is_open:
        return None

    try:
        line = ser.readline().decode(errors="ignore").strip()

        if not line:
            return None

        if line.startswith("DATA:"):
            values = line.replace("DATA:", "").split(",")

            if len(values) == 5:
                return {
                    "temperature": float(values[0]),
                    "humidity": float(values[1]),
                    "light": float(values[2]),
                    "air": float(values[3]),
                    "co2": float(values[4])
                }

    except Exception as e:
        print("❌ Read error:", e)

    return None


# ---------------------------
# 🔹 START
# ---------------------------
if __name__ == "__main__":
    init_serial()

    while True:
        data = get_sensor_data()
        if data:
            print("📊", data)
        time.sleep(1)