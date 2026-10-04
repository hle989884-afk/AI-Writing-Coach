
import json
import logging
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from google import genai
from google.genai import types

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

MAX_PROMPT_LENGTH = 10000
MAX_INSTRUCTION_LENGTH = 10000

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    index_file = BASE_DIR / "index.html"

    if not index_file.is_file():
        return jsonify({
            "error": "Không tìm thấy index.html trên máy chủ."
        }), 500

    return send_from_directory(BASE_DIR, "index.html")


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "AI Writing Coach"
    }), 200


# --------------------------------------------------
# GEMINI AI GENERATION
# --------------------------------------------------

@app.route("/api/generate", methods=["POST"])
def generate():
    # 1. Check API key
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if not api_key:
        app.logger.error("GEMINI_API_KEY is not configured.")
        return jsonify({
            "error": "Máy chủ chưa được cấu hình Gemini API key."
        }), 500

    # 2. Check request content type
    if not request.is_json:
        return jsonify({
            "error": "Yêu cầu phải gửi dữ liệu JSON."
        }), 415

    # 3. Parse incoming JSON
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Dữ liệu gửi lên không hợp lệ."
        }), 400

    prompt = data.get("prompt", "")
    system_instruction = data.get("system_instruction", "")

    if not isinstance(prompt, str) or not prompt.strip():
        return jsonify({
            "error": "Vui lòng nhập nội dung bài viết."
        }), 400

    if not isinstance(system_instruction, str):
        return jsonify({
            "error": "System instruction không hợp lệ."
        }), 400

    if len(prompt) > MAX_PROMPT_LENGTH:
        return jsonify({
            "error": "Nội dung bài viết quá dài."
        }), 400

    if len(system_instruction) > MAX_INSTRUCTION_LENGTH:
        return jsonify({
            "error": "Hướng dẫn AI quá dài."
        }), 400

    # 4. Call Gemini
    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )

        result_text = response.text

        if not result_text or not result_text.strip():
            app.logger.warning("Gemini returned an empty response.")
            return jsonify({
                "error": "AI không trả về kết quả. Vui lòng thử lại."
            }), 502

        # 5. Validate Gemini's JSON before sending it to frontend
        try:
            parsed_result = json.loads(result_text)
        except json.JSONDecodeError:
            app.logger.error(
                "Gemini returned invalid JSON: %s",
                result_text[:500]
            )
            return jsonify({
                "error": (
                    "AI trả về dữ liệu không đúng định dạng. "
                    "Vui lòng thử lại."
                )
            }), 502

        # Keep compatibility with the existing frontend:
        # callGeminiAPI() expects data.text to contain JSON text.
        return jsonify({
            "text": json.dumps(
                parsed_result,
                ensure_ascii=False
            )
        }), 200

    except Exception as exc:
        # Log technical details on Render, but never expose secrets.
        app.logger.exception(
            "Gemini API request failed (%s)",
            type(exc).__name__
        )

        error_name = type(exc).__name__.lower()
        error_message = str(exc).lower()

        if (
            "unauthorized" in error_name
            or "api key not valid" in error_message
            or "invalid api key" in error_message
            or "permission_denied" in error_message
        ):
            return jsonify({
                "error": (
                    "Gemini API key không hợp lệ hoặc chưa có "
                    "quyền sử dụng API."
                )
            }), 502

        if (
            "resource_exhausted" in error_name
            or "429" in error_message
        ):
            return jsonify({
                "error": (
                    "Gemini API đã đạt giới hạn sử dụng. "
                    "Vui lòng thử lại sau."
                )
            }), 429

        return jsonify({
            "error": (
                "Không thể xử lý yêu cầu AI lúc này. "
                "Vui lòng kiểm tra cấu hình và Render Logs."
            )
        }), 502


# --------------------------------------------------
# ERROR HANDLERS
# --------------------------------------------------

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Không tìm thấy đường dẫn yêu cầu."
    }), 404


@app.errorhandler(413)
def request_too_large(error):
    return jsonify({
        "error": "Dữ liệu gửi lên quá lớn."
    }), 413


@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        "error": "Máy chủ gặp lỗi nội bộ."
    }), 500


# --------------------------------------------------
# START SERVER
# --------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
