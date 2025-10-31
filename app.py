import os
import time
import requests
from flask import Flask, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Replicate API Token
REPLICATE_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# Upload folders
UPLOAD_DIR = "uploads"
OUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


# 🏠 Serve the frontend (index.html)
@app.route("/")
def home():
    return send_from_directory("static", "index.html")


# 🧠 Enhance endpoint (main AI function)
@app.route("/api/enhance", methods=["POST"])
def enhance():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    # Save uploaded image
    image_file = request.files["image"]
    filename = secure_filename(image_file.filename)
    input_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{filename}")
    image_file.save(input_path)

    # Prepare API request to Replicate (GFPGAN model)
    headers = {"Authorization": f"Token {REPLICATE_TOKEN}"}
    data = {
        "version": "928944f3d8892e5484d6a8b8c8b2c975b8027c35a1be5a9837e8c8f9a7e64c59",  # GFPGAN v1.4
        "input": {"img": open(input_path, "rb")}
    }

    # Step 1: Create prediction
    print("📤 Sending image to Replicate...")
    create_response = requests.post(
        "https://api.replicate.com/v1/predictions",
        headers=headers,
        json={
            "version": data["version"],
            "input": {
                # We must upload to a public URL first or use base64 if hosted privately.
                # Here, we use Replicate's direct upload.
                "img": f"https://replicate.delivery/pbxt/{os.path.basename(input_path)}"
            }
        }
    )

    if create_response.status_code != 201:
        print(create_response.text)
        return jsonify({"error": "Failed to create prediction"}), 500

    prediction = create_response.json()
    prediction_id = prediction.get("id")

    # Step 2: Poll for completion
    print("⏳ Waiting for prediction result...")
    output_url = None
    for _ in range(30):  # max ~30 seconds
        status_response = requests.get(
            f"https://api.replicate.com/v1/predictions/{prediction_id}",
            headers=headers
        )
        result = status_response.json()
        if result["status"] == "succeeded":
            output_url = result["output"][0]
            break
        elif result["status"] == "failed":
            return jsonify({"error": "Prediction failed"}), 500
        time.sleep(2)

    if not output_url:
        return jsonify({"error": "No output from Replicate"}), 500

    # Step 3: Download enhanced image
    print("📥 Downloading enhanced image...")
    output_path = os.path.join(OUT_DIR, f"enh_{filename}")
    img_data = requests.get(output_url).content
    with open(output_path, "wb") as f:
        f.write(img_data)

    return send_file(output_path, mimetype="image/jpeg")


# 🧾 Health check route
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "message": "Replicate enhancer ready!"})


# 🧩 Serve static files (optional)
@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


# 🚀 Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
