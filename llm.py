import os
import json

# ---------------------------
# CI MODE DETECTION
# ---------------------------
CI_MODE = os.getenv("CI") == "true"

# ---------------------------
# SAFE IMPORTS
# ---------------------------
if not CI_MODE:
    try:
        from groq import Groq
        from dotenv import load_dotenv

        load_dotenv()
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    except:
        client = None
else:
    client = None

MODEL = "llama-3.1-8b-instant"


# ---------------------------
# MAIN FUNCTION
# ---------------------------
def generate_explanation(data, ml_prediction):

    # ---------------------------
    # CI / FALLBACK MODE
    # ---------------------------
    if client is None:
        return {
            "prediction": ml_prediction,
            "reasons": [
                "AI disabled (CI mode or missing API key)"
            ],
            "suggestions": [
                "Run locally with GROQ_API_KEY for AI insights"
            ]
        }

    # ---------------------------
    # PROMPT
    # ---------------------------
    prompt = f"""
You are an INDOOR ROOM COMFORT ANALYSIS SYSTEM.

STRICT RULES (MUST FOLLOW):
1. Use ONLY the given values.
2. DO NOT invent or assume any numbers.
3. DO NOT mention any value not present in DATA.
4. This is INDOOR environment (not weather).
5. Keep reasoning short and logical.
6. If no clear issue: say "Combined environmental factors may reduce comfort".

DATA:
Temperature: {data['temperature']} °C
Humidity: {data['humidity']} %
Light: {data['light']} lux
AQI: {data['air']}
CO2: {data['co2']} ppm

Prediction: {ml_prediction}

OUTPUT FORMAT (STRICT JSON ONLY):

{{
    "prediction": "{ml_prediction}",
    "reasons": [
        "reason 1",
        "reason 2"
    ],
    "suggestions": [
        "suggestion 1",
        "suggestion 2"
    ]
}}
"""

    # ---------------------------
    # API CALL
    # ---------------------------
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a strict data-grounded AI. Never hallucinate values."},
                {"role": "user", "content": prompt}
            ],
            model=MODEL,
            temperature=0.1
        )

        content = response.choices[0].message.content.strip()

        # ---------------------------
        # SAFE PARSE
        # ---------------------------
        try:
            parsed = json.loads(content)
            return parsed
        except:
            return {
                "prediction": ml_prediction,
                "reasons": ["Model output parsing failed"],
                "suggestions": ["Retry or check system"]
            }

    except Exception as e:
        return {
            "prediction": ml_prediction,
            "reasons": [f"AI Error: {str(e)}"],
            "suggestions": []
        }


# ---------------------------
# WRAPPER FUNCTION
# ---------------------------
def generate_ai_insight(data, ml_prediction):
    return generate_explanation(data, ml_prediction)