"""
inFeedo GEO Content Generator
Flask backend — serves the UI and handles API calls for trending topics + content generation.

Run: python app.py
Open: http://localhost:5000
"""

import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from trending import get_trending_topics
from generator import generate_geo_content

load_dotenv()

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/trending", methods=["POST"])
def trending():
    """
    Fetch top 10 trending AI queries for a given sector + pain point.
    Body: { sector: str, pain_point: str }
    """
    data       = request.get_json(force=True)
    sector     = data.get("sector", "IT")
    pain_point = data.get("pain_point", "Predicting Attrition")

    if not os.getenv("ANTHROPIC_API_KEY"):
        return jsonify({"error": "ANTHROPIC_API_KEY not set in .env"}), 500

    try:
        topics = get_trending_topics(sector, pain_point)
        return jsonify({"topics": topics})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate", methods=["POST"])
def generate():
    """
    Generate GEO-optimised content.
    Body: { platform, tone, sector, pain_point, target_query }
    """
    data         = request.get_json(force=True)
    platform     = data.get("platform", "Blog")
    tone         = data.get("tone", "Informative with Data")
    sector       = data.get("sector", "IT")
    pain_point   = data.get("pain_point", "Predicting Attrition")
    target_query = data.get("target_query", "")

    if not target_query:
        return jsonify({"error": "target_query is required"}), 400

    if not os.getenv("ANTHROPIC_API_KEY"):
        return jsonify({"error": "ANTHROPIC_API_KEY not set in .env"}), 500

    try:
        result = generate_geo_content(
            platform=platform,
            tone=tone,
            sector=sector,
            pain_point=pain_point,
            target_query=target_query,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"\n inFeedo GEO Content Generator running at http://localhost:{port}\n")
    app.run(debug=True, port=port)
