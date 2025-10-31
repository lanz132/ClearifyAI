import os
import requests
import replicate
from flask import Flask, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()
app = Flask(__name__)

# Replicate API key
REPLICATE_TOKEN = os.getenv("REPLICATE_API_TOKEN")
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_TOKEN

# Directories
UPLOAD_DIR = "uploads"
OUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


# 🏠 Serve index.html from static folder
@app.route("/")
def home():
    return send_from_directory("static", "index.html")


# ⚙️ Main Enhance Endpoint
@app.route("/api/enhance", methods=["POST"])
def enhance():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    f = request.files["image"]
    filename = secure_filename(f.filename)
    input_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{filename}")
    f.save(input_path)

    try:
        print("🧠 Step 1: Face restoration using GFPGAN...")
        gfpgan_output = replicate.run(
            "tencentarc/gfpgan:latest",
            input={"img": open(input_path, "rb")}
        )

        if not gfpgan_output or len(gfpgan_output) == 0:
            return jsonify({"error": "GFPGAN failed to produce output"}), 500

        face_fixed_url = gfpgan_output[0]

        print("🔍 Step 2: HD upscaling using Real-ESRGAN...")
        realesrgan_output = replicate.run(
            "xinntao/realesrgan:latest",
            input={"img": face_fixed_url}
        )

        if not realesrgan_output or len(realesrgan_output) == 0:
            return jsonify({"error": "Real-ESRGAN failed to produce output"}), 500

        enhanced_url = realesrgan_output[0]

        print("📥 Downloading enhanced image...")
        output_path = os.path.join(OUT_DIR, f"enh_{filename}")
        img_data = requests.get(enhanced_url).content
        with open(output_path, "wb") as out_f:
            out_f.write(img_data)

        print("✅ Enhancement complete.")
        return send_file(output_path, mimetype="image/jpeg")

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500


# 🧾 Health Check
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "message": "Replicate enhancer combo ready!"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
