import os
import requests
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import uuid

load_dotenv()
app = Flask(__name__)

REPLICATE_TOKEN = os.getenv("REPLICATE_API_TOKEN")

UPLOAD_DIR = "uploads"
OUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

@app.route("/api/enhance", methods=["POST"])
def enhance():
    if 'image' not in request.files:
        return jsonify({"error": "No image file"}), 400

    f = request.files['image']
    filename = secure_filename(f.filename)
    input_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{filename}")
    f.save(input_path)

    # Upload the image to Replicate
    print("Sending to Replicate API...")
    headers = {"Authorization": f"Token {REPLICATE_TOKEN}"}
    json_data = {
        "version": "928944f3d8892e5484d6a8b8c8b2c975b8027c35a1be5a9837e8c8f9a7e64c59",  # GFPGAN v1.4
        "input": {"img": open(input_path, "rb")}
    }

    response = requests.post(
        "https://api.replicate.com/v1/predictions",
        headers=headers,
        json={
            "version": json_data["version"],
            "input": {
                "img": f"https://replicate.delivery/pbxt/{os.path.basename(input_path)}"
            }
        }
    )

    if response.status_code != 201:
        return jsonify({"error": response.text}), 500

    prediction = response.json()
    return jsonify(prediction)

@app.route("/")
def index():
    return jsonify({"status": "ok", "message": "Replicate enhancer ready!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
