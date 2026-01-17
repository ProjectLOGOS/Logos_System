#!/usr/bin/env python3
"""
Backend server for PXL Interactive Web Demo.

Provides REST API endpoints for proposition analysis.
Uses Flask for HTTP server and calls Coq for verification.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    from flask import Flask, jsonify, request, send_from_directory
except ImportError as exc:  # pragma: no cover - configuration error
    raise RuntimeError(
        "Flask is required for the interactive web demo. Install it via "
        "'pip install flask'."
    ) from exc

# Add parent directory to path for imports
DEMOS_DIR = Path(__file__).parent
REPO_ROOT = DEMOS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

BASELINE_DIR = REPO_ROOT / "Protopraxis" / "formal_verification" / "coq" / "baseline"

app = Flask(__name__, static_folder=".")

# Enable CORS for development


@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "GET, POST")
    return response


@app.route("/")
def index():
    """Serve the main HTML page."""
    return send_from_directory(".", "index.html")


@app.route("/<path:path>")
def static_files(path):
    """Serve static files."""
    return send_from_directory(".", path)


@app.route("/api/analyze", methods=["POST"])
def analyze_proposition():
    """
    Analyze a proposition using PXL logic.

    Request JSON:
        {
            "proposition": "P \\/ ~P",
            "mode": "quick" | "full"
        }

    Response JSON:
        {
            "success": true,
            "proposition": "...",
            "grounding": "...",
            "truthValue": "...",
            "analysis": [...],
            "coqOutput": "..." (if mode=full)
        }
    """
    try:
        data = request.get_json(force=True)
    except TypeError:
        return jsonify({"success": False, "error": "Invalid JSON payload"}), 400

    proposition = str(data.get("proposition", "")).strip()
    mode = data.get("mode", "quick")

    if not proposition:
        return jsonify({"success": False, "error": "No proposition provided"}), 400

    if mode == "quick":
        result = quick_analysis(proposition)
        return jsonify(result)

    if mode == "full":
        result = full_coq_analysis(proposition)
        return jsonify(result)

    return jsonify({"success": False, "error": f"Unknown mode: {mode}"}), 400


def quick_analysis(proposition):
    """
    Quick pattern-based analysis without Coq.
    Matches common patterns and provides instant feedback.
    """
    result = {
        "success": True,
        "proposition": proposition,
        "mode": "quick",
        "analysis": [],
    }

    # LEM detection
    if "P \\/ ~P" in proposition or "P ∨ ¬P" in proposition:
        result["grounding"] = "𝕀₁ (constructively proven)"
        result["truthValue"] = "TRUE (theorem)"
        result["analysis"] = [
            "✅ Law of Excluded Middle detected",
            "This is a proven theorem in PXL",
            "Derived from trinitarian decidability",
            "Zero extra assumptions required",
        ]

    # Liar paradox
    elif "false" in proposition.lower() and "statement" in proposition.lower():
        result["grounding"] = "UNGROUNDED"
        result["truthValue"] = "INDETERMINATE"
        result["paradoxStatus"] = "Resolved via privative negation"
        result["analysis"] = [
            "⚠️ Self-referential paradox detected",
            "Cannot ground in 𝕀₁ (truth) - circular",
            "Cannot ground in 𝕀₂ (falsity) - circular",
            "✅ Resolution: Statement is metaphysically incoherent",
        ]

    # Modal necessity
    elif "□" in proposition or "coherence(𝕆)" in proposition:
        result["grounding"] = "𝕀₃ (modal anchor)"
        result["truthValue"] = "NECESSARY"
        result["modalProperties"] = ["Necessity (□)"]
        result["analysis"] = [
            "✅ Modal necessity claim",
            "coherence(𝕆) is axiomatically necessary",
            "Grounds in third identity anchor (𝕀₃)",
            "Provides foundation for all truths",
        ]

    # Generic
    else:
        result["grounding"] = "To be determined"
        result["truthValue"] = "Requires Coq verification"
        result["analysis"] = [
            "ℹ️ Generic proposition",
            "Use mode=full for complete Coq verification",
            "Would check grounding in 𝕀₁, 𝕀₂, or 𝕀₃",
        ]

    return result


def full_coq_analysis(proposition):
    """
    Full analysis using Coq verification.
    Attempts to prove/disprove the proposition.
    """
    # This would require encoding proposition as Coq syntax
    # For now, return a stub indicating the feature
    return {
        "success": True,
        "proposition": proposition,
        "mode": "full",
        "grounding": "Coq verification pending",
        "truthValue": "PENDING",
        "analysis": [
            "ℹ️ Full Coq verification requires:",
            "  1. Encoding proposition in Coq syntax",
            "  2. Attempting proof construction",
            "  3. Checking assumption footprint",
            "",
            "💡 For now, use the command-line demos:",
            "  python3 demos/alignment_demo.py",
            "  python3 demos/lem_demo.py",
        ],
        "coqOutput": "(Feature in development)",
    }


@app.route("/api/status", methods=["GET"])
def status():
    """Check backend status."""
    return jsonify(
        {
            "status": "online",
            "backend": "Python Flask",
            "coq_available": check_coq_available(),
            "kernel": "8-axiom minimal",
        }
    )


def check_coq_available():
    """Check if Coq is installed and accessible."""
    try:
        result = subprocess.run(
            ["coqc", "--version"],
            capture_output=True,
            timeout=2,
            check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PXL interactive demo server")
    parser.add_argument(
        "--demo-ok",
        action="store_true",
        help="Acknowledge starting the interactive demo server (requires LOGOS_OPERATOR_OK=1)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    """Run the Flask development server."""
    args = parse_args(argv)
    if os.environ.get("LOGOS_OPERATOR_OK", "").strip() != "1":
        print("ERROR: LOGOS_OPERATOR_OK=1 is required to start the demo server.", file=sys.stderr)
        return 2
    if not args.demo_ok:
        print("ERROR: --demo-ok is required to start the demo server.", file=sys.stderr)
        return 2
    print("=" * 70)
    print("  PXL Interactive Web Demo - Backend Server")
    print("=" * 70)
    print("\nStarting server...")
    print(f"Backend directory: {DEMOS_DIR}")
    print(f"Coq available: {check_coq_available()}")
    print("\n🌐 Open your browser to: http://localhost:5000")
    print("=" * 70)

    # Change to demo directory so static files work
    os.chdir(DEMOS_DIR)

    app.run(host="0.0.0.0", port=5000, debug=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
