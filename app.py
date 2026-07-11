import os
import logging
import torch

from flask import Flask, render_template, request, jsonify
from transformers import pipeline

# ======================================================
# Flask Application
# ======================================================

app = Flask(__name__)

# ======================================================
# Logging Configuration
# ======================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("GrammarGuardian")

# ======================================================
# Model Configuration
# ======================================================

MODEL_NAME = "vennify/t5-base-grammar-correction"

grammar_model = None
MODEL_READY = False

# ======================================================
# Load Grammar Correction Model
# ======================================================
def load_model():
    """
    Load Hugging Face Grammar Correction Model.
    """

    global grammar_model, MODEL_READY

    try:

        logger.info("Loading Grammar Guardian Model...")

        device = 0 if torch.cuda.is_available() else -1

        grammar_model = pipeline(
            task="text2text-generation",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            device=device
        )

        MODEL_READY = True

        logger.info("Grammar model loaded successfully.")

    except Exception as e:

        MODEL_READY = False

        logger.error("Model loading failed.")
        logger.error(str(e))

# Load model when application starts
load_model()

# ======================================================
# Text Preprocessing
# ======================================================

def preprocess_text(text):
    """
    Clean input text before grammar correction.
    """

    if text is None:
        return ""

    text = text.strip()

    text = " ".join(text.split())

    return text


# ======================================================
# Grammar Correction Function
# ======================================================

def correct_grammar(text):
    """
    Correct grammar using the T5 model.
    """

    if not MODEL_READY:

        return {
            "success": False,
            "message": "Grammar model is unavailable.",
            "corrected": ""
        }

    text = preprocess_text(text)

    if text == "":

        return {
            "success": False,
            "message": "Input text is empty.",
            "corrected": ""
        }

    try:

        prompt = "grammar: " + text

        result = grammar_model(
            prompt,
            max_length=256,
            num_beams=4,
            early_stopping=True
        )

        corrected_text = result[0]["generated_text"]

        return {

            "success": True,

            "original": text,

            "corrected": corrected_text

        }

    except Exception as e:

        logger.error(str(e))

        return {

            "success": False,

            "message": str(e),

            "corrected": ""

        }
    # ======================================================
# Home Page
# ======================================================

@app.route("/")
def home():
    return render_template("index.html")


# ======================================================
# About Page
# ======================================================

@app.route("/about")
def about():
    return render_template("about.html")


# ======================================================
# Grammar Correction API
# ======================================================

@app.route("/correct", methods=["POST"])
def correct():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "success": False,
                "message": "No data received."
            }), 400

        text = data.get("text", "")

        result = correct_grammar(text)

        if not result["success"]:

            return jsonify(result), 400

        corrected_text = result["corrected"]

        return jsonify({

            "success": True,

            "original": result["original"],

            "corrected": corrected_text,

            "word_count": len(corrected_text.split()),

            "character_count": len(corrected_text),

            "message": "Grammar corrected successfully."

        })

    except Exception as e:

        logger.exception(e)

        return jsonify({

            "success": False,

            "message": "Internal Server Error"

        }), 500


# ======================================================
# Health Check
# ======================================================

@app.route("/health")
def health():

    return jsonify({

        "status": "Running",

        "application": "Grammar Guardian",

        "model": MODEL_NAME,

        "model_loaded": MODEL_READY

    })


# ======================================================
# Error Handlers
# ======================================================

@app.errorhandler(404)
def page_not_found(error):

    return render_template("about.html"), 404


@app.errorhandler(500)
def internal_server_error(error):

    return jsonify({

        "success": False,

        "message": "Internal Server Error"

    }), 500
# ======================================================
# Application Information
# ======================================================

@app.route("/info")
def info():

    return jsonify({

        "project": "Grammar Guardian",

        "version": "1.0.0",

        "framework": "Flask",

        "model": MODEL_NAME,

        "python": "3.11",

        "developer": "Bhargavi Bhatraju"

    })


# ======================================================
# Main
# ======================================================

if __name__ == "__main__":

    logger.info("=" * 50)
    logger.info("Grammar Guardian Started Successfully")
    logger.info("=" * 50)

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host=host,
        port=port,
        debug=True
    )