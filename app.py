
import os
from pathlib import Path
from flask import Flask, send_from_directory, jsonify

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)

@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)