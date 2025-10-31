import os
import requests
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import uuid

load_dotenv()

app = Flask(__name__)
DEEP_AI_KEY = os.getenv("DEEP_AI_KEY")

UPLOAD_DIR = "uploads"
OUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

@app.route("/api/enhance", methods=["POST"])
def enhance():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    f = request.files['image']
    filename = secure_filename(f.filename)
    input_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{filename}")
    f.save(input_path)

    r = requests.post(
        "https://api.deepai.org/api/torch-srgan",
        files={'image': open(input_path, 'rb')},
        headers={'api-key': DEEP_AI_KEY}
    )

    if r.status_code != 200:
        return jsonify({"error": r.text}), 500

    data = r.json()
    out_url = data.get("output_url")
    if not out_url:
        return jsonify({"error": "DeepAI returned no image"}), 500

    # Download the enhanced image
    out_path = os.path.join(OUT_DIR, f"enh_{filename}")
    img = requests.get(out_url)
    with open(out_path, "wb") as f2:
        f2.write(img.content)

    return send_file(out_path, mimetype="image/jpeg")

@app.route("/")
def index():
    return app.send_static_file("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)