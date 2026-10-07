import json
import logging
import os
import random
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from google import genai
from google.genai import types

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")
API_KEY = (
    os.environ.get("GEMINI_API_KEY")
    or os.environ.get("GOOGLE_API_KEY")
)

if API_KEY:
    logger.info("Gemini API key detected.")
else:
    logger.error("Gemini API key NOT detected.")
# Gemini 3.8 Flash is the primary model requested by the API.
PRIMARY_MODEL = "gemini-3.5-flash"

FALLBACK_MODELS = [
    "gemini-3.8-flash-lite"
]

MAX_OUTPUT_TOKENS = int(
    os.environ.get("MAX_OUTPUT_TOKENS", "4096")
)

# Number of extra retries after the first attempt.
MAX_RETRIES_PER_MODEL = int(
    os.environ.get("GEMINI_RETRIES", "2")
)

RETRY_BASE_SECONDS = float(
    os.environ.get("GEMINI_RETRY_BASE", "1")
)

PORT = int(os.environ.get("PORT", "5000"))

# ============================================================
# APP / LOGGING
# ============================================================

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger("AI-Writing-Coach")

client = None

if API_KEY:
    client = genai.Client(api_key=API_KEY)
    logger.info("Gemini client initialized.")

    try:
        available_models = []

        for model in client.models.list():
            supported = getattr(model, "supported_actions", []) or []

            if "generateContent" in supported:
                name = model.name.replace("models/", "")
                available_models.append(name)

        logger.info("Available Gemini models: %s", available_models)

    except Exception as e:
        logger.exception("Cannot list Gemini models: %s", e)

else:
    logger.error(
        "GEMINI_API_KEY / GOOGLE_API_KEY is not configured."
    )


# ============================================================
# MODEL HELPERS
# ============================================================

def get_model_chain():
    """Return unique primary + fallback models."""
    models = [PRIMARY_MODEL] + FALLBACK_MODELS

    result = []
    for model in models:
        model = str(model).strip()
        if model and model not in result:
            result.append(model)

    return result


def get_error_status(error):
    """Try to extract an HTTP status code from a Gemini exception."""
    for attr in ("code", "status_code", "http_status"):
        value = getattr(error, attr, None)

        if isinstance(value, int):
            return value

        if isinstance(value, str) and value.isdigit():
            return int(value)

    text = str(error)

    for code in (
        400,
        401,
        403,
        404,
        408,
        409,
        429,
        500,
        502,
        503,
        504,
    ):
        if str(code) in text:
            return code

    return None


def is_retryable_error(error):
    """Retry only transient errors."""
    status = get_error_status(error)

    if status in (408, 429, 500, 502, 503, 504):
        return True

    text = str(error).lower()

    retry_words = [
        "unavailable",
        "overloaded",
        "temporarily unavailable",
        "service unavailable",
        "timeout",
        "timed out",
        "rate limit",
        "resource exhausted",
        "internal server error",
        "bad gateway",
        "503",
        "429",
        "500",
        "502",
        "504",
    ]

    return any(word in text for word in retry_words)


def error_message(error):
    text = str(error).strip()

    if text:
        return text[:1500]

    return error.__class__.__name__


# ============================================================
# GEMINI GENERATION
# ============================================================

def generate_with_fallback(system_instruction, prompt):
    if client is None:
        raise RuntimeError(
            "GEMINI_API_KEY chưa được cấu hình trên Render."
        )

    models = get_model_chain()
    last_error = None

    for model_index, model_name in enumerate(models):

        logger.info(
            "Trying model %s (%d/%d)",
            model_name,
            model_index + 1,
            len(models),
        )

        for attempt in range(MAX_RETRIES_PER_MODEL + 1):

            try:
                # Gemini 3.8 Flash:
                # Do not send temperature/top_p/top_k here.
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                    ),
                )

                result_text = response.text

                if not result_text or not result_text.strip():
                    raise RuntimeError(
                        f"{model_name} trả về kết quả rỗng."
                    )

                logger.info(
                    "Gemini success: model=%s attempt=%d",
                    model_name,
                    attempt + 1,
                )

                return result_text, model_name

            except Exception as error:
                last_error = error
                status = get_error_status(error)

                logger.exception(
                    "Gemini error: model=%s attempt=%d status=%s",
                    model_name,
                    attempt + 1,
                    status,
                )

                # 401/403/404 are not transient.
                # Do not waste retries on authentication,
                # permission, or unavailable-model errors.
                if (
                    is_retryable_error(error)
                    and attempt < MAX_RETRIES_PER_MODEL
                ):
                    delay = (
                        RETRY_BASE_SECONDS * (2 ** attempt)
                        + random.uniform(0, 0.35)
                    )

                    logger.warning(
                        "Retrying %s in %.2f seconds.",
                        model_name,
                        delay,
                    )

                    time.sleep(delay)
                    continue

                # Move to fallback model.
                if model_index < len(models) - 1:
                    next_model = models[model_index + 1]

                    logger.warning(
                        "Model %s failed. Falling back to %s.",
                        model_name,
                        next_model,
                    )

                break

    raise last_error or RuntimeError(
        "Tất cả Gemini model đều thất bại."
    )


# ============================================================
# JSON HELPERS
# ============================================================

def normalize_model_json(text):
    """Remove accidental markdown fences and validate JSON."""
    if not isinstance(text, str):
        raise json.JSONDecodeError(
            "Gemini response is not text.",
            str(text),
            0,
        )

    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    parsed = json.loads(text)

    return json.dumps(
        parsed,
        ensure_ascii=False,
        separators=(",", ":"),
    )


# ============================================================
# ROUTES
# ============================================================

@app.route("/", methods=["GET"])
def index():
    index_file = BASE_DIR / "index.html"

    if not index_file.exists():
        return jsonify(
            {
                "error": "Không tìm thấy index.html.",
                "base_dir": str(BASE_DIR),
            }
        ), 404

    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "AI Writing Coach",
            "model": PRIMARY_MODEL,
            "fallback_models": FALLBACK_MODELS,
            "api_key_configured": bool(API_KEY),
        }
    )


@app.route("/api/generate", methods=["POST"])
def generate():
    if not request.is_json:
        return jsonify(
            {
                "error": "Request phải có Content-Type: application/json."
            }
        ), 400

    body = request.get_json(silent=True)

    if not isinstance(body, dict):
        return jsonify(
            {
                "error": "JSON request không hợp lệ."
            }
        ), 400

    system_instruction = body.get(
        "system_instruction",
        "",
    )

    prompt = body.get(
        "prompt",
        "",
    )

    if not isinstance(system_instruction, str):
        system_instruction = str(system_instruction)

    if not isinstance(prompt, str):
        prompt = str(prompt)

    if not prompt.strip():
        return jsonify(
            {
                "error": "Prompt không được để trống."
            }
        ), 400

    try:
        result_text, used_model = generate_with_fallback(
            system_instruction=system_instruction,
            prompt=prompt,
        )

        normalized = normalize_model_json(result_text)

        return jsonify(
            {
                "text": normalized,
                "model": used_model,
            }
        ), 200

    except json.JSONDecodeError:
        logger.exception(
            "Gemini returned invalid JSON."
        )

        return jsonify(
            {
                "error": "Gemini trả về dữ liệu không phải JSON hợp lệ.",
            }
        ), 502

    except Exception as error:
        status = get_error_status(error)
        details = error_message(error)

        logger.exception(
            "All Gemini attempts failed."
        )

        if status == 401:
            message = (
                "Gemini API authentication thất bại. "
                "Kiểm tra GEMINI_API_KEY trên Render."
            )

        elif status == 403:
            message = (
                "Gemini API key không có quyền sử dụng model này."
            )

        elif status == 404:
            message = (
                "Model Gemini không khả dụng cho API key này."
            )

        elif status == 429:
            message = (
                "Gemini API đang giới hạn quota. "
                "Đã thử retry và model fallback."
            )

        elif status in (500, 502, 503, 504):
            message = (
                "Gemini đang tạm thời không khả dụng. "
                "Đã thử retry và model fallback."
            )

        else:
            message = (
                "Không thể xử lý yêu cầu AI."
            )

        return jsonify(
            {
                "error": message,
                "details": details,
            }
        ), 502


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    logger.info(
        "Starting AI Writing Coach on port %s",
        PORT,
    )

    logger.info(
        "Primary model: %s",
        PRIMARY_MODEL,
    )

    logger.info(
        "Fallback models: %s",
        FALLBACK_MODELS,
    )

    logger.info(
        "API key configured: %s",
        bool(API_KEY),
    )

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
    )
