import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from google import genai
from google.genai import types


# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

# Maximum HTTP request body
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

app.logger.setLevel(logging.INFO)


# --------------------------------------------------
# Gemini configuration
# --------------------------------------------------

GEMINI_MODEL = os.environ.get(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
).strip()

# Maximum characters accepted from frontend.
# Characters != tokens, but this prevents abnormally large requests.
MAX_PROMPT_LENGTH = 10000
MAX_INSTRUCTION_LENGTH = 6000

# Output token limit.
# For a middle-school writing coach, 800-1200 is usually enough.
MAX_OUTPUT_TOKENS = int(
    os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "1000")
)

TEMPERATURE = 0.2


# --------------------------------------------------
# Retry configuration
# --------------------------------------------------

MAX_RETRIES = 2

RETRY_DELAYS = (
    1.5,
    3.0,
)


# ==================================================
# RESPONSE CACHE
# ==================================================

# Cache is extremely useful for an education app.
#
# Example:
#
# Student sends:
# "I like play football."
#
# Then presses Analyze again.
#
# The second request can be served from cache
# without calling Gemini again.

CACHE_MAX_ITEMS = 100

CACHE_TTL_SECONDS = 10 * 60  # 10 minutes

_cache = OrderedDict()

_cache_lock = threading.Lock()


def _make_cache_key(prompt: str, system_instruction: str) -> str:
    """
    Create a stable SHA-256 key.

    We don't store the actual essay as the dictionary key,
    which keeps memory usage more predictable.
    """

    raw = (
        system_instruction.strip()
        + "\n---PROMPT---\n"
        + prompt.strip()
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def _get_cached_result(cache_key: str):
    """
    Return cached result if it exists and hasn't expired.
    """

    now = time.time()

    with _cache_lock:

        item = _cache.get(cache_key)

        if item is None:
            return None

        created_at, result = item

        # Expired
        if now - created_at > CACHE_TTL_SECONDS:
            _cache.pop(cache_key, None)
            return None

        # Move recently used item to the end
        _cache.move_to_end(cache_key)

        return result


def _set_cached_result(cache_key: str, result):
    """
    Store result in a small LRU-style cache.
    """

    with _cache_lock:

        _cache[cache_key] = (
            time.time(),
            result
        )

        _cache.move_to_end(cache_key)

        while len(_cache) > CACHE_MAX_ITEMS:
            _cache.popitem(last=False)


# ==================================================
# GEMINI CLIENT
# ==================================================

# Reuse the client instead of creating it for every request.
#
# This does NOT magically increase Gemini quota,
# but it reduces unnecessary initialization overhead.

GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY",
    ""
).strip()

client = None

if GEMINI_API_KEY:

    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        app.logger.info(
            "Gemini client initialized successfully."
        )

    except Exception:
        app.logger.exception(
            "Failed to initialize Gemini client."
        )


# ==================================================
# HOME PAGE
# ==================================================

@app.route("/", methods=["GET"])
def home():

    index_file = BASE_DIR / "index.html"

    if not index_file.is_file():

        return jsonify({
            "error": "Không tìm thấy index.html trên máy chủ."
        }), 500

    return send_from_directory(
        BASE_DIR,
        "index.html"
    )


# ==================================================
# HEALTH CHECK
# ==================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "service": "AI Writing Coach",
        "model": GEMINI_MODEL
    }), 200


# ==================================================
# GEMINI GENERATION
# ==================================================

@app.route("/api/generate", methods=["POST"])
def generate():

    # --------------------------------------------------
    # 1. Check Gemini client
    # --------------------------------------------------

    if client is None:

        return jsonify({
            "error": (
                "Máy chủ chưa được cấu hình Gemini API key "
                "hoặc Gemini client chưa khởi tạo được."
            )
        }), 500


    # --------------------------------------------------
    # 2. Check request type
    # --------------------------------------------------

    if not request.is_json:

        return jsonify({
            "error": "Yêu cầu phải gửi dữ liệu JSON."
        }), 415


    # --------------------------------------------------
    # 3. Parse JSON
    # --------------------------------------------------

    data = request.get_json(silent=True)

    if not isinstance(data, dict):

        return jsonify({
            "error": "Dữ liệu gửi lên không hợp lệ."
        }), 400


    prompt = data.get(
        "prompt",
        ""
    )

    system_instruction = data.get(
        "system_instruction",
        ""
    )


    # --------------------------------------------------
    # 4. Validate prompt
    # --------------------------------------------------

    if not isinstance(prompt, str):

        return jsonify({
            "error": "Nội dung bài viết không hợp lệ."
        }), 400


    prompt = prompt.strip()

    if not prompt:

        return jsonify({
            "error": "Vui lòng nhập nội dung bài viết."
        }), 400


    if len(prompt) > MAX_PROMPT_LENGTH:

        return jsonify({
            "error": (
                f"Nội dung bài viết quá dài. "
                f"Tối đa {MAX_PROMPT_LENGTH} ký tự."
            )
        }), 400


    # --------------------------------------------------
    # 5. Validate system instruction
    # --------------------------------------------------

    if not isinstance(system_instruction, str):

        return jsonify({
            "error": "System instruction không hợp lệ."
        }), 400


    system_instruction = system_instruction.strip()


    if len(system_instruction) > MAX_INSTRUCTION_LENGTH:

        return jsonify({
            "error": (
                f"System instruction quá dài. "
                f"Tối đa {MAX_INSTRUCTION_LENGTH} ký tự."
            )
        }), 400


    # ==================================================
    # CACHE CHECK
    # ==================================================

    cache_key = _make_cache_key(
        prompt,
        system_instruction
    )

    cached_result = _get_cached_result(
        cache_key
    )

    if cached_result is not None:

        app.logger.info(
            "Cache hit - Gemini request skipped."
        )

        return jsonify({
            "text": cached_result
        }), 200


    app.logger.info(
        "Cache miss - sending request to Gemini."
    )


    # ==================================================
    # GEMINI REQUEST
    # ==================================================

    try:

        response = None

        for attempt in range(MAX_RETRIES + 1):

            try:

                response = client.models.generate_content(

                    model=GEMINI_MODEL,

                    contents=prompt,

                    config=types.GenerateContentConfig(

                        system_instruction=system_instruction,

                        response_mime_type="application/json",

                        temperature=TEMPERATURE,

                        max_output_tokens=MAX_OUTPUT_TOKENS,
                    ),
                )

                break


            except Exception as exc:

                error_text = str(exc).upper()

                is_service_unavailable = (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                )


                # --------------------------------------------------
                # IMPORTANT:
                # Do NOT retry quota errors.
                #
                # Retrying 429 usually wastes more time and
                # does not solve the quota problem.
                # --------------------------------------------------

                if not is_service_unavailable:

                    raise


                if attempt >= MAX_RETRIES:

                    raise


                wait_seconds = RETRY_DELAYS[
                    min(
                        attempt,
                        len(RETRY_DELAYS) - 1
                    )
                ]

                app.logger.warning(
                    "Gemini temporarily unavailable. "
                    "Retry %s/%s after %.1fs.",
                    attempt + 1,
                    MAX_RETRIES,
                    wait_seconds
                )

                time.sleep(
                    wait_seconds
                )


        # --------------------------------------------------
        # Safety check
        # --------------------------------------------------

        if response is None:

            return jsonify({
                "error": (
                    "AI không trả về phản hồi. "
                    "Vui lòng thử lại."
                )
            }), 502


        # --------------------------------------------------
        # Extract text
        # --------------------------------------------------

        result_text = response.text


        if not result_text:

            app.logger.warning(
                "Gemini returned empty response."
            )

            return jsonify({
                "error": (
                    "AI không trả về kết quả. "
                    "Vui lòng thử lại."
                )
            }), 502


        result_text = result_text.strip()


        # ==================================================
        # VALIDATE JSON
        # ==================================================

        try:

            parsed_result = json.loads(
                result_text
            )

        except json.JSONDecodeError:

            app.logger.error(
                "Gemini returned invalid JSON."
            )

            return jsonify({
                "error": (
                    "AI trả về dữ liệu không đúng "
                    "định dạng JSON. Vui lòng thử lại."
                )
            }), 502


        # ==================================================
        # NORMALIZE RESULT
        # ==================================================

        normalized_result = json.dumps(
            parsed_result,
            ensure_ascii=False,
            separators=(",", ":")
        )


        # ==================================================
        # SAVE TO CACHE
        # ==================================================

        _set_cached_result(
            cache_key,
            normalized_result
        )


        # ==================================================
        # FRONTEND COMPATIBILITY
        # ==================================================

        return jsonify({
            "text": normalized_result
        }), 200


    # ==================================================
    # ERROR HANDLING
    # ==================================================

    except Exception as exc:

        app.logger.exception(
            "Gemini API request failed (%s)",
            type(exc).__name__
        )

        error_name = type(exc).__name__.lower()
        error_message = str(exc).lower()


        # --------------------------------------------------
        # 503
        # --------------------------------------------------

        if (
            "503" in error_message
            or "unavailable" in error_message
        ):

            return jsonify({
                "error": (
                    "AI đang bận hoặc quá tải. "
                    "Vui lòng đợi một chút rồi thử lại."
                )
            }), 503


        # --------------------------------------------------
        # 429 - QUOTA
        # --------------------------------------------------

        if (
            "resource_exhausted" in error_name
            or "429" in error_message
            or "quota" in error_message
        ):

            return jsonify({
                "error": (
                    "Gemini API đã đạt giới hạn quota. "
                    "Vui lòng thử lại sau."
                )
            }), 429


        # --------------------------------------------------
        # Authentication
        # --------------------------------------------------

        if (
            "unauthorized" in error_name
            or "api key not valid" in error_message
            or "invalid api key" in error_message
            or "permission_denied" in error_message
            or "permission denied" in error_message
        ):

            return jsonify({
                "error": (
                    "Gemini API key không hợp lệ "
                    "hoặc chưa có quyền sử dụng API."
                )
            }), 502


        # --------------------------------------------------
        # Generic error
        # --------------------------------------------------

        return jsonify({
            "error": (
                "Không thể xử lý yêu cầu AI lúc này. "
                "Vui lòng kiểm tra Render Logs."
            )
        }), 502


# ==================================================
# ERROR HANDLERS
# ==================================================

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


# ==================================================
# START SERVER
# ==================================================

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
    )ss
