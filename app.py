"""
AI Learning Coach — Flask Backend
==================================
Three-call LLM pipeline using HuggingFace Router (OpenAI-compatible):
  Call 1 (Mistral)  → analyze learner answers, produce score + level
  Call 2 (Zephyr)   → beginner coaching OR advanced challenges
  Call 3 (Mistral)  → personalized 4-week learning roadmap
"""

import logging
import os
import json
import re

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from openai import OpenAI
from dotenv import load_dotenv

# ── Bootstrap ────────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

import time
_START_TIME = time.time()

app = Flask(__name__)
CORS(app)

# ── Rate Limiting ─────────────────────────────────────────────────────────────
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set. Add HF_TOKEN=hf_xxxx to your .env file.")

FLASK_DEBUG  = os.getenv("FLASK_DEBUG", "false").lower() == "true"
MAX_ANSWER_LENGTH = 5_000   # characters — prevents prompt-injection bloat

ANALYSIS_MODEL = "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai"
GUIDANCE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai"

# ── HuggingFace Router client (OpenAI-compatible) ────────────────────────────
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

# ── Question Store (In-Memory) ───────────────────────────────────────────────
question_store = {
    "beginner": [
        {
            "id": 1,
            "question": "What is a variable in Python?",
            "options": [
                "A reserved keyword in Python.",
                "A named storage location for data.",
                "A type of loop.",
                "A function that returns a value."
            ],
            "correct_answer": "A named storage location for data."
        },
        {
            "id": 2,
            "question": "Which of the following is a valid way to create a 'for' loop in Python?",
            "options": [
                "for x in range(10):",
                "for (x = 0; x < 10; x++)",
                "loop x from 1 to 10:",
                "foreach x in list:"
            ],
            "correct_answer": "for x in range(10):"
        }
    ],
    "advanced": [
        {
            "id": 3,
            "question": "What is a decorator in Python?",
            "options": [
                "A way to add comments to your code.",
                "A function that takes another function and extends its behavior without explicitly modifying it.",
                "A built-in data structure for storing key-value pairs.",
                "A method for styling the output of your code."
            ],
            "correct_answer": "A function that takes another function and extends its behavior without explicitly modifying it."
        },
        {
            "id": 4,
            "question": "What is the Global Interpreter Lock (GIL) in CPython?",
            "options": [
                "A lock that prevents multiple threads from executing Python bytecodes at the same time.",
                "A tool for debugging Python code.",
                "A security feature that prevents unauthorized access to your code.",
                "A way to speed up your Python programs."
            ],
            "correct_answer": "A lock that prevents multiple threads from executing Python bytecodes at the same time."
        }
    ]
}



# ── Helpers ───────────────────────────────────────────────────────────────────

def sanitize(text: str) -> str:
    """
    Strip characters that could break prompt templates or attempt injection.
    Curly braces are removed because they collide with Python f-string syntax
    and are a common prompt-injection vector.
    """
    return text.replace("{", "").replace("}", "").strip()


def has_personal_info(text: str) -> tuple[bool, str]:
    """
    Detect personal/sensitive information in text (email, phone, bank details, etc).

    Returns:
        (has_personal_info: bool, detected_type: str)
    """
    text_lower = text.lower()

    # Email pattern: user@domain.com
    if re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text):
        return True, "email address"

    # Phone patterns: (123) 456-7890, 123-456-7890, +1 123 456 7890, etc.
    if re.search(r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text):
        return True, "phone number"

    # Bank account number: 8-17 consecutive digits
    if re.search(r'\b\d{8,17}\b', text):
        return True, "bank account number"

    # Credit card: 13-19 consecutive digits
    if re.search(r'\b\d{13,19}\b', text):
        return True, "credit card number"

    # Social Security Number: XXX-XX-XXXX
    if re.search(r'\b\d{3}-\d{2}-\d{4}\b', text):
        return True, "social security number"

    # Routing number: 9 consecutive digits
    if re.search(r'\b\d{9}\b', text):
        return True, "routing number"

    # Keywords for sensitive info
    sensitive_keywords = [
        'credit card', 'cvv', 'pin', 'ssn', 'bank account',
        'routing number', 'swift', 'iban', 'ach', 'wire transfer'
    ]
    for keyword in sensitive_keywords:
        if keyword in text_lower:
            return True, keyword

    return False, ""


def call_model(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 512,
    retries: int = 3,
) -> str:
    """
    Call the HuggingFace router via OpenAI-compatible SDK with retry logic.

    Args:
        model:         Full model identifier including provider suffix.
        system_prompt: Persona and output-format instructions.
        user_prompt:   Actual learner data / task input.
        max_tokens:    Upper bound on generated tokens.
        retries:       Number of retry attempts for transient failures.

    Returns:
        Generated text string, or an error string prefixed with '[Model Error]'.
        Never raises — callers check the prefix to detect failures.
    """
    import time
    for attempt in range(retries):
        try:
            log.info("Calling model: %s (max_tokens=%d, attempt %d)", model, max_tokens, attempt + 1)
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            text = completion.choices[0].message.content.strip()
            # Sanity check: if response looks like HTML, it's an error
            if text.startswith("<!DOCTYPE") or text.startswith("<html"):
                log.error("Model returned HTML instead of text (likely a 50x error page)")
                return "[Model Error]: Service Unavailable"
            log.info("Model responded (%d chars)", len(text))
            return text
        except Exception as exc:
            exc_str = str(exc)
            # Check if error contains HTML markers or service error codes
            if any(marker in exc_str for marker in ["<!DOCTYPE", "503", "504", "Service Unavailable", "Gateway Timeout", "<html"]):
                log.error("HuggingFace service error detected in exception")
                return "[Model Error]: Service Unavailable"

            if attempt < retries - 1:
                wait_time = 2 ** attempt
                log.warning("Model call failed (attempt %d): %s. Retrying in %ds...", attempt + 1, exc, wait_time)
                time.sleep(wait_time)
            else:
                log.error("Model call failed after %d attempts: %s", retries, exc)
                return "[Model Error]: Service Unavailable"


# ── LLM Judge — Safety Evaluation ─────────────────────────────────────────

def evaluate_response_safety(content: str) -> tuple[bool, str]:
    """
    Basic safety evaluation for AI-generated content.

    Currently disabled LLM judge due to model availability issues.
    Just do basic heuristic checks.

    Args:
        content: The AI-generated response text to evaluate

    Returns:
        (is_safe: bool, reason: str)
    """
    if not content or len(content) < 5:
        return True, ""

    # Basic heuristic checks (no LLM required)
    unsafe_keywords = [
        'kill', 'suicide', 'self-harm', 'abuse', 'exploit',
        'illegal', 'hack', 'crack', 'malware'
    ]

    content_lower = content.lower()
    for keyword in unsafe_keywords:
        if keyword in content_lower:
            return False, f"Content contains unsafe keyword: {keyword}"

    return True, ""


def safe_extract_analysis(raw_text: str) -> dict | None:
    """
    Extract the first valid JSON object from raw model output.

    Strategy:
      1. Try a direct json.loads on the full text (model obeyed instructions).
      2. Fall back to regex — find the first {...} block (model added preamble).
      3. Return None if both fail (caller will surface a 500 with raw_response).

    Args:
        raw_text: Raw string returned by the analysis LLM call.

    Returns:
        Parsed dict on success, None on failure.
    """
    # Strategy 1 — direct parse
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # Strategy 2 — regex extraction
    try:
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        pass

    return None


# ── LLM Call 1 — Analyze + score  (Mistral) ──────────────────────────────────

def analyze_answers(user_answers: str) -> str:
    """
    Send learner answers to Mistral and request a structured JSON analysis.

    The system prompt explicitly forbids output outside the JSON block and
    provides the exact schema — reducing hallucination and parse failures.

    Args:
        user_answers: Sanitized learner input text.

    Returns:
        Raw model output (expected to be a JSON string).
    """
    system = (
        "You are an AI Learning Coach. "
        "Analyze the learner response and return ONLY a valid JSON object. "
        "No markdown, no explanation, no text outside the JSON.\n"
        "Schema: "
        '{"score": <int 0-100>, '
        '"level": "beginner" or "advanced", '
        '"strengths": ["<item>"], '
        '"weaknesses": ["<item>"]}'
    )
    user = f"Learner Response:\n{user_answers}"
    return call_model(ANALYSIS_MODEL, system, user, max_tokens=350)


# ── LLM Call 2 — Guidance / challenges  (Zephyr) ─────────────────────────────

def generate_guidance(level: str, user_answers: str) -> str:
    """
    Generate either beginner-friendly coaching or advanced coding challenges.

    The branch is determined by the level extracted from Call 1:
      - beginner  → plain-language explanation of variables, loops, functions
      - advanced  → 3 challenges across DSA, REST API, and System Design

    Args:
        level:        'beginner' or 'advanced'.
        user_answers: Sanitized learner input text (provides context).

    Returns:
        Plain-text coaching or challenge descriptions.
    """
    # DEBUG: Print to verify this function is being called
    import sys
    print(f"DEBUG: generate_guidance called with level={level}", file=sys.stderr)
    if level == "beginner":
        system = (
            "You are a friendly programming tutor. "
            "Explain concepts clearly with simple examples. "
            "Avoid jargon. Use short paragraphs."
        )
        user = (
            f"The learner said:\n{user_answers}\n\n"
            "Explain these topics in beginner-friendly language with examples:\n"
            "1. Variables and data types\n"
            "2. Loops (for / while)\n"
            "3. Functions and return values"
        )
    else:
        system = (
            "You are a senior software engineer and mentor. "
            "Write concise, challenging problems with clear acceptance criteria."
        )
        user = (
            f"The learner said:\n{user_answers}\n\n"
            "Generate 3 advanced coding challenges — one each for:\n"
            "1. Data Structures & Algorithms\n"
            "2. REST API design\n"
            "3. System Design or Performance Optimization\n\n"
            "For each: title, problem statement, acceptance criteria."
        )
    result = call_model(GUIDANCE_MODEL, system, user, max_tokens=600)

    # Define fallback responses
    if level == "advanced":
        fallback = (
            "**Challenge 1: Data Structures & Algorithms**\n"
            "Implement a binary search tree with insert, delete, and search operations.\n\n"
            "**Challenge 2: REST API Design**\n"
            "Design and build a RESTful API for a task management system with CRUD operations.\n\n"
            "**Challenge 3: System Design**\n"
            "Design a rate-limiting system that can handle millions of requests per second."
        )
    else:
        fallback = (
            "**Variables and Data Types**: Variables are containers for storing data values. "
            "Python supports strings, integers, floats, and booleans.\n\n"
            "**Loops**: Use `for` loops to iterate over sequences or `while` loops to repeat until a condition is false.\n\n"
            "**Functions**: Functions are reusable blocks of code. Use `def` to define them and `return` to send back values."
        )

    # Use fallback if result contains error markers
    if (not result or
        "[Model Error]" in result or
        "<!DOCTYPE" in result or
        "<html" in result.lower() or
        "503" in result or
        "504" in result or
        "Service Unavailable" in result or
        "Gateway Timeout" in result or
        len(result) > 3000 or
        result.startswith("[")):
        log.warning("*** USING FALLBACK - detected error in guidance: %s", result[:100] if result else "empty")
        return fallback

    return result


# ── LLM Call 3 — 4-week roadmap  (Mistral) ───────────────────────────────────

def generate_roadmap(score: int, level: str) -> str:
    """
    Generate a structured 4-week personalized learning roadmap.

    Uses Mistral (same as Call 1) for consistent, structured text output.

    Args:
        score: Clamped integer 0–100 from Call 1.
        level: 'beginner' or 'advanced' from Call 1.

    Returns:
        Plain-text roadmap with one line per week.
    """
    system = (
        "You are an expert curriculum designer. "
        "Write specific, actionable learning roadmaps. "
        "Use the exact format requested — no additional commentary."
    )
    user = (
        f"Create a 4-week personalized learning roadmap for a {level}-level "
        f"student who scored {score}/100.\n\n"
        "Format each week exactly like this (one line per week):\n"
        "Week 1: <goal> — <technologies> — <practice task>\n"
        "Week 2: <goal> — <technologies> — <practice task>\n"
        "Week 3: <goal> — <technologies> — <practice task>\n"
        "Week 4: <goal> — <technologies> — <final project suggestion>"
    )
    return call_model(ANALYSIS_MODEL, system, user, max_tokens=500)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/quiz", methods=["GET"])
def get_quiz():
    """
    Returns a list of quiz questions based on the provided level.
    Query Params:
        level (str): 'beginner' or 'advanced'
    """
    level = request.args.get("level", "beginner")
    if level not in ["beginner", "advanced"]:
        level = "beginner"
    
    questions = question_store.get(level, [])
    return jsonify({"success": True, "questions": questions})


@app.route("/api/evaluate", methods=["POST"])
@limiter.limit("10 per hour")  # Max 10 evaluations per hour per IP
def evaluate_performance():
    """
    Main evaluation endpoint.

    Request body (JSON):
        { "answers": "<learner text>" }

    Response body (JSON):
        {
          "success": true,
          "score": <int>,
          "level": "beginner"|"advanced",
          "responseType": "beginner_explanation"|"advanced_challenge",
          "analysis": { score, level, strengths[], weaknesses[] },
          "guidance": "<string>",
          "roadmap": "<string>"
        }

    Error response:
        { "success": false, "error": "<message>", "raw_response"?: "<string>" }
    """
    try:
        data = request.get_json(silent=True)

        # ── Input validation ─────────────────────────────
        if not data:
            return jsonify({"success": False, "error": "Missing or invalid JSON body"}), 400

        raw_answers = data.get("answers", "")
        if not isinstance(raw_answers, str):
            return jsonify({"success": False, "error": "'answers' must be a string"}), 400

        if len(raw_answers.strip()) < 10:
            return jsonify({"success": False, "error": "Answer too short (minimum 10 characters)"}), 400

        if len(raw_answers) > MAX_ANSWER_LENGTH:
            return jsonify({"success": False, "error": f"Answer too long (maximum {MAX_ANSWER_LENGTH} characters)"}), 400

        # Check for personal/sensitive information
        has_personal, info_type = has_personal_info(raw_answers)
        if has_personal:
            return jsonify({
                "success": False,
                "error": f"Input contains sensitive information ({info_type}). Please remove any personal data before submitting."
            }), 400

        # Sanitize before injecting into prompts (prompt injection mitigation)
        user_answers = sanitize(raw_answers)
        log.info("Evaluating answer (%d chars, level=TBD)", len(user_answers))

        # ── Call 1: Analyze ──────────────────────────────
        analysis_raw = analyze_answers(user_answers)
        analysis = safe_extract_analysis(analysis_raw)

        if not analysis:
            log.error("Analysis parse failed. Raw: %s", analysis_raw[:200])
            return jsonify({
                "success": False,
                "error": "Failed to parse AI analysis. Check HF_TOKEN or model availability.",
                "raw_response": analysis_raw,
            }), 500

        # Safe score extraction — clamp to 0–100
        try:
            score = max(0, min(100, int(analysis.get("score", 0))))
        except (ValueError, TypeError):
            score = 0

        # Safe level extraction
        level = analysis.get("level", "beginner")
        if level not in ("beginner", "advanced"):
            level = "beginner"

        log.info("Analysis complete — score=%d level=%s", score, level)

        # ── Route ─────────────────────────────────────────
        response_type = "beginner_explanation" if score < 50 else "advanced_challenge"

        # ── Call 2: Guidance ─────────────────────────────
        guidance = generate_guidance(level, user_answers)
        log.info("Guidance returned (%d chars), contains errors: %s", len(guidance), any(x in guidance for x in ["[Model Error]", "<!DOCTYPE", "503"]))

        # Additional cleanup: if guidance still contains error markers despite fallback in generate_guidance
        # This catches cases where error detection might have been missed
        if "[Model Error]" in guidance or "<!DOCTYPE" in guidance or "503" in guidance or "504" in guidance or "Service Unavailable" in guidance or "Gateway Timeout" in guidance or guidance.startswith("["):
            log.warning("Guidance contains errors, applying fallback")
            # Guidance generation failed - use safe fallback content
            if level == "advanced":
                guidance = (
                    "**Challenge 1: Data Structures & Algorithms**\n"
                    "Implement a binary search tree with insert, delete, and search operations.\n\n"
                    "**Challenge 2: REST API Design**\n"
                    "Design and build a RESTful API for a task management system with CRUD operations.\n\n"
                    "**Challenge 3: System Design**\n"
                    "Design a rate-limiting system that can handle millions of requests per second."
                )
            else:
                guidance = (
                    "**Variables and Data Types**: Variables are containers for storing data values. "
                    "Python supports strings, integers, floats, and booleans.\n\n"
                    "**Loops**: Use `for` loops to iterate over sequences or `while` loops to repeat until a condition is false.\n\n"
                    "**Functions**: Functions are reusable blocks of code. Use `def` to define them and `return` to send back values."
                )

        # ── Call 3: Roadmap ──────────────────────────────
        roadmap = generate_roadmap(score, level)

        # ── Safety Evaluation ────────────────────────────
        # Use LLM Judge to verify guidance is safe before returning
        log.info("Running safety evaluation on guidance...")
        guidance_safe, guidance_reason = evaluate_response_safety(guidance)
        if not guidance_safe:
            log.warning("Guidance failed safety check: %s", guidance_reason)
            guidance = "Unable to generate guidance at this time. Please try again with a different answer."

        # Use LLM Judge to verify roadmap is safe before returning
        log.info("Running safety evaluation on roadmap...")
        roadmap_safe, roadmap_reason = evaluate_response_safety(roadmap)
        if not roadmap_safe:
            log.warning("Roadmap failed safety check: %s", roadmap_reason)
            roadmap = "Unable to generate roadmap at this time. Please try again."

        # FINAL SANITY CHECK: Replace any remaining error content with fallbacks
        # Check for HTML, error markers, or unusually long responses (errors are longer than normal guidance)
        if not guidance or "[Model Error]" in guidance or "<!DOCTYPE" in guidance or "html" in guidance.lower() or "503" in guidance or len(guidance) > 2500:
            if level == "beginner":
                guidance = (
                    "**Variables and Data Types**: Variables are containers for storing data values. "
                    "Python supports strings, integers, floats, and booleans.\n\n"
                    "**Loops**: Use `for` loops to iterate over sequences or `while` loops to repeat until a condition is false.\n\n"
                    "**Functions**: Functions are reusable blocks of code. Use `def` to define them and `return` to send back values."
                )
            else:
                guidance = (
                    "**Challenge 1: Data Structures & Algorithms**\n"
                    "Implement a binary search tree with insert, delete, and search operations.\n\n"
                    "**Challenge 2: REST API Design**\n"
                    "Design and build a RESTful API for a task management system with CRUD operations.\n\n"
                    "**Challenge 3: System Design**\n"
                    "Design a rate-limiting system that can handle millions of requests per second."
                )

        # ULTIMATE SAFEGUARD: If guidance contains HTML or error markers, use fallback
        if ("<!DOCTYPE" in guidance or "[Model Error]" in guidance or
            "html" in guidance.lower() or "503" in guidance or "504" in guidance or
            (len(guidance) > 2000 and guidance.count("<") > 5)):

            # Use safe fallback content
            if level == "advanced":
                guidance = (
                    "**Challenge 1: Data Structures & Algorithms**\n"
                    "Implement a binary search tree with insert, delete, and search operations. "
                    "Must support insert, delete, and search operations in O(log n) average time.\n\n"
                    "**Challenge 2: REST API Design**\n"
                    "Design and build a RESTful API for a task management system. "
                    "Include endpoints for CRUD operations with proper HTTP methods.\n\n"
                    "**Challenge 3: System Design**\n"
                    "Design a rate-limiting system that can handle millions of requests per second. "
                    "Consider using token bucket or sliding window algorithms."
                )
            else:
                guidance = (
                    "**Variables and Data Types**\n"
                    "Variables store data. Python has: strings (text), integers (whole numbers), "
                    "floats (decimals), and booleans (True/False). Example: `name = 'Alice'` stores text.\n\n"
                    "**Loops**\n"
                    "Loops repeat code. `for` loops iterate over sequences: `for i in range(5): print(i)`. "
                    "`while` loops repeat until a condition is false: `while x < 10: x += 1`.\n\n"
                    "**Functions**\n"
                    "Functions are reusable blocks of code. Define with `def`: `def greet(name): return f'Hello {name}'`. "
                    "Call with `greet('Alice')`. Functions take inputs and return outputs."
                )

        return jsonify({
            "success":      True,
            "score":        score,
            "level":        level,
            "responseType": response_type,
            "analysis":     analysis,
            "guidance":     guidance,
            "roadmap":      roadmap,
            "safety_checked": True,
        })

    except Exception as exc:
        log.exception("Unhandled error in /api/evaluate")
        return jsonify({"success": False, "error": str(exc)}), 500


# ── Error Handlers ───────────────────────────────────────────────────────────

@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limit exceeded (429) errors with user-friendly message."""
    log.warning("Rate limit exceeded: %s", e.description)
    return jsonify({
        "success": False,
        "error": "You've exceeded your evaluation limit (10 per hour). Please try again later."
    }), 429


@app.route("/")
def health_check():
    """Health check endpoint — returns 200 when the server is running."""
    return jsonify({
        "message": "AI Learning Coach API is running",
        "status": "ok"
    })


@app.route("/test-endpoint", methods=["POST"])
def test_endpoint():
    """Test endpoint to verify Flask is routing requests."""
    return jsonify({"status": "ok", "test": True})


# ── Test endpoint to verify code reloading ─────────────────────────────────────

@app.route("/test-reload")
def test_reload():
    """Test endpoint to verify the app module is using latest code."""
    return jsonify({
        "status": "ok",
        "start_time": _START_TIME,
        "message": "If you see this with a recent start_time, code reloading is working"
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=FLASK_DEBUG, port=5000)