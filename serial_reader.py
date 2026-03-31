import serial

ser = serial.Serial('COM8', 9600, timeout=1)

def get_sensor_data():
    try:
        line = ser.readline().decode().strip()

        if line.startswith("DATA:"):
            line = line.replace("DATA:", "")
            values = line.split(",")

            if len(values) == 5:
                return {
                    "temperature": float(values[0]),
                    "humidity": float(values[1]),
                    "light": float(values[2]),
                    "air": float(values[3]),
                    "co2": float(values[4])
                }

    except Exception as e:
        print("Error:", e)

    return None