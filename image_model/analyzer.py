"""
image_model/analyzer.py
Image threat-analysis logic — sends an uploaded PIL image to Google Gemini
and returns a structured threat-assessment response.
"""

import streamlit as st

try:
    from google import genai
except ImportError:
    genai = None

try:
    from PIL import Image
except ImportError:
    Image = None

# Gemini model to use for image analysis
_GEMINI_MODEL = "gemini-2.0-flash"

_SYSTEM_PROMPT = (
    "You are a forest patrol assistant. Analyze the image and answer in JSON format "
    "with the following schema:\n"
    "{\n"
    '  "threat_found": <true|false>,\n'
    '  "confidence": <0-1 float>,\n'
    '  "message": <short summary>,\n'
    '  "details": <optional longer explanation>\n'
    "}\n"
    "If the image shows obvious illegal activity (logging, hunting, poaching, trespassing), "
    "set threat_found to true. Otherwise set it to false.\n"
    "Only output valid JSON; do not include extra text.\n"
)


def analyze_image(image: "Image.Image") -> str:
    """
    Send *image* to Gemini and return the raw JSON response text.

    Returns an error string (not JSON) if the request fails.
    """
    if genai is None:
        return (
            "google-genai is not installed. "
            "Install it with `pip install google-genai` and restart the app."
        )
    if Image is None:
        return (
            "Pillow is not installed. "
            "Install it with `pip install pillow` and restart the app."
        )

    try:
        client   = genai.Client()
        response = client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=[_SYSTEM_PROMPT, image],
        )
        return response.text
    except Exception as e:
        return f"Error querying Gemini: {e}"


def is_available() -> bool:
    """Return True if both google-genai and Pillow are installed."""
    return genai is not None and Image is not None
