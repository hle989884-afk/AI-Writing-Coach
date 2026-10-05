import json
import logging
import os
import time
from pathlib import Path
from collections import OrderedDict
import hashlib
import threading

from flask import Flask, jsonify, request, send_from_directory
from google import genai
from google.genai import types


# ==============================
# CONFIG
# ==============================

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

MODEL_NAME = os.environ.get(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

MAX_PROMPT_LENGTH = 10000
MAX_INSTRUCTION_LENGTH = 6000
MAX_OUTPUT_TOKENS = 1000

MAX_RETRIES = 2


# ==============================
# GEMINI CLIENT
# ==============================

API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

client = None

if API_KEY:
    try:
        client = genai.Client(api_key=API_KEY)
        app.logger.info("Gemini client initialized.")
    except Exception:
        app.logger.exception("Could not initialize Gemini client.")


# ==============================
# CACHE
# ==============================

CACHE_MAX_ITEMS = 100
CACHE_TTL = 600

cache = OrderedDict()
cache_lock = threading.Lock()


def make_cache_key(prompt, instruction):
    text = instruction.strip() + "\n---\n" + prompt.strip()

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def get_cache(key):
    now = time.time()

    with cache_lock:
        item = cache.get(key)

        if item is None:
            return None

        created_time, value = item

        if now - created_time > CACHE_TTL:
            cache.pop(key, None)
            return None

        cache.move_to_end(key)

        return value


def set_cache(key, value):
    with cache_lock:
        cache[key] = (time.time(), value)
        cache.move_to_end(key)

        while len(cache) > CACHE_MAX_ITEMS:
            cache.popitem(last=False)


# ==============================
# HOME
# ==============================

@app.route("/", methods=["GET"])
def home():

    index_file = BASE_DIR / "index.html"

    if not index_file.is_file():
        return jsonify({
            "error": "Không tìm thấy index.html."
        }), 500

    return send_from_directory(
        BASE_DIR,
        "index.html"
    )


# ==============================
# HEALTH CHECK
# ==============================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "service": "AI Writing Coach",
        "model": MODEL_NAME
    }), 200


# ==============================
# GENERATE
# ==============================

@app.route("/api/generate", methods=["POST"])
def generate():

    # Check API key
    if client is None:
        return jsonify({
            "error": "Gemini API key chưa được cấu hình."
        }), 500

    # Check JSON
    if not request.is_json:
        return jsonify({
            "error": "Yêu cầu phải là JSON."
        }), 415

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Dữ liệu không hợp lệ."
        }), 400

    prompt = data.get("prompt", "")
    system_instruction = data.get(
        "system_instruction",
        ""
    )

    # Validate prompt
    if not isinstance(prompt, str):
        return jsonify({
            "error": "Prompt không hợp lệ."
        }), 400

    prompt = prompt.strip()

    if not prompt:
        return jsonify({
            "error": "Vui lòng nhập bài viết."
        }), 400

    if len(prompt) > MAX_PROMPT_LENGTH:
        return jsonify({
            "error": "Bài viết quá dài."
        }), 400

    # Validate system instruction
    if not isinstance(system_instruction, str):
        return jsonify({
            "error": "System instruction không hợp lệ."
        }), 400

    system_instruction = system_instruction.strip()

    if len(system_instruction) > MAX_INSTRUCTION_LENGTH:
        return jsonify({
            "error": "System instruction quá dài."
        }), 400

    # ==============================
    # CACHE
    # ==============================

    cache_key = make_cache_key(
        prompt,
        system_instruction
    )

    cached = get_cache(cache_key)

    if cached is not None:

        app.logger.info(
            "Cache hit. Gemini request skipped."
        )

        return jsonify({
            "text": cached
        }), 200

    # ==============================
    # GEMINI
    # ==============================

    response = None

    try:

        for attempt in range(MAX_RETRIES + 1):

            try:

                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        temperature=0.2,
                        max_output_tokens=MAX_OUTPUT_TOKENS
                    )
                )

                break

            except Exception as exc:

                error_text = str(exc).upper()

                is_unavailable = (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                )

                if not is_unavailable:
                    raise

                if attempt >= MAX_RETRIES:
                    raise

                wait_time = 2 ** attempt

                app.logger.warning(
                    "Gemini unavailable. Retry in %s seconds.",
                    wait_time
                )

                time.sleep(wait_time)

        if response is None:
            return jsonify({
                "error": "Gemini không trả về phản hồi."
            }), 502

        result_text = response.text

        if not result_text:
            return jsonify({
                "error": "AI không trả về kết quả."
            }), 502

        result_text = result_text.strip()

        # ==============================
        # VALIDATE JSON
        # ==============================

        try:
            parsed = json.loads(result_text)

        except json.JSONDecodeError:

            app.logger.error(
                "Gemini returned invalid JSON."
            )

            return jsonify({
                "error": "AI trả về JSON không hợp lệ."
            }), 502

        normalized = json.dumps(
            parsed,
            ensure_ascii=False,
            separators=(",", ":")
        )

        # Save cache
        set_cache(
            cache_key,
            normalized
        )

        # Keep frontend compatibility
        return jsonify({
            "text": normalized
        }), 200

    except Exception as exc:

        app.logger.exception(
            "Gemini request failed: %s",
            type(exc).__name__
        )

        error_text = str(exc).lower()
        error_name = type(exc).__name__.lower()

        # 429 / quota
        if (
            "429" in error_text
            or "quota" in error_text
            or "resource_exhausted" in error_name
        ):
            return jsonify({
                "error": (
                    "Gemini API đã đạt giới hạn quota."
                )
            }), 429

        # 503
        if (
            "503" in error_text
            or "unavailable" in error_text
        ):
            return jsonify({
                "error": (
                    "Gemini đang quá tải. "
                    "Vui lòng thử lại sau."
                )
            }), 503

        # API key
        if (
            "api key" in error_text
            or "unauthorized" in error_text
            or "permission" in error_text
        ):
            return jsonify({
                "error": (
                    "Gemini API key không hợp lệ "
                    "hoặc không có quyền."
                )
            }), 502

        return jsonify({
            "error": (
                "Không thể xử lý yêu cầu AI."
            )
        }), 502


# ==============================
# ERROR HANDLERS
# ==============================

@app.errorhandler(404)
def not_found(error):

    return jsonify({
        "error": "Không tìm thấy đường dẫn."
    }), 404


@app.errorhandler(413)
def request_too_large(error):

    return jsonify({
        "error": "Dữ liệu gửi lên quá lớn."
    }), 413


@app.errorhandler(500)
def internal_error(error):

    return jsonify({
        "error": "Server gặp lỗi nội bộ."
    }), 500


# ==============================
# START
# ==============================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            "5000"
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
