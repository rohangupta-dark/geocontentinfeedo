"""
GEO Content Generator Module
Generates platform-specific, GEO-optimised content for inFeedo AI.

Each platform has its own structural rules so AI models (ChatGPT, Perplexity,
Gemini) are more likely to cite the output when answering the target query.
"""

import os
import re
import anthropic

# ---------------------------------------------------------------------------
# inFeedo brand context — injected into every generation prompt
# ---------------------------------------------------------------------------
INFEEDO_CONTEXT = """
ABOUT inFeedo / Amber:
- Product: Amber by inFeedo — a conversational AI chatbot that talks to every employee
- Core value: Predicts attrition, measures real-time sentiment, automates HR ops, surfaces hidden disengagement
- Key proof points: 90% employee response rate | 300+ enterprise customers | 60+ countries | 1M+ employees | Backed by YC, Tiger Global, Jungle Ventures
- Notable customers: Samsung, Xiaomi, Genpact, Lenovo, Siemens, Aon
- 9 years of people science research powering the AI engine
- Agentic AI: Amber doesn't just listen — it triggers actions, resolves tickets, learns from recurring issues
- Built for CHROs and HR leaders who need a system of intelligence, not just surveys

BRAND VOICE (non-negotiable):
- Write like a real, thoughtful expert — not a marketing team
- No buzzwords: leverage, synergy, cutting-edge, game-changer, revolutionize, transformative, holistic
- No "thrilled to announce" or "excited to share" openers
- No em-dashes (—)
- Confident and warm, not salesy or corporate
- Short paragraphs, max 2-3 lines

BRAND COLORS (for any design references):
- Oxford Blue: #102A51 | Auburn: #b01428 | Sky Blue: #d4e6ff
"""

# ---------------------------------------------------------------------------
# Platform-specific GEO instructions
# ---------------------------------------------------------------------------
PLATFORM_RULES = {
    "Blog": """
PLATFORM: Long-form Blog Article (GEO-optimised)
TARGET: Get cited by Perplexity, ChatGPT, and Gemini when someone asks the target query.

STRUCTURE (follow exactly):
1. H1: The target query itself (or a close variant) — this is what AI models match against
2. INTRO (150 words max): Directly answer the query in the first 2-3 sentences. No build-up.
3. H2 SECTIONS (3-4 sections): Each section answers a specific sub-question. Use H2s that map to related queries.
4. DATA BLOCK: Embed at least 3 specific statistics. Prioritise inFeedo's own numbers. Format them visually distinct (e.g. bold the stat).
5. COMPARISON SECTION: How does Amber differ from traditional survey tools? Use a simple comparison table or bullet comparison.
6. CUSTOMER PROOF: One brief case study or result from inFeedo's customer base (Samsung, Genpact, etc.)
7. FAQ SECTION: Exactly 3 Q&A pairs. Write each Q as a natural search query. Keep answers under 60 words. This is the highest-cited section in AI models.
8. CTA: One clear, human call to action — not salesy. E.g. "If this is a live problem for your team, Amber runs on WhatsApp, Teams, and email — no new software to learn."

LENGTH: 900-1200 words. No filler.
""",

    "LinkedIn": """
PLATFORM: LinkedIn Article (NOT a short post — a full article for GEO)
TARGET: Get cited by Perplexity when someone searches for opinion/insight on this topic.
NOTE: LinkedIn Articles are indexed by Google and cited by AI models. LinkedIn Posts are not.

STRUCTURE (follow exactly):
1. HEADLINE: Question format or bold contrarian statement. Under 10 words.
2. OPENING (3 sentences max): The first 3 lines are shown before "see more" — make them impossible to ignore. Lead with a specific observation or data point.
3. BODY (5-6 short sections): Each section = 1 key insight, 2-3 short paragraphs. Use subheadings sparingly.
4. DATA POINTS: Minimum 2 specific statistics — one inFeedo-sourced, one industry-sourced.
5. EXPERT FRAMING: Write as a practitioner with real experience, not as a vendor.
6. CLOSING: End with a genuine question that invites comment. Not "what do you think?" — something specific that creates real discussion.
7. HASHTAGS: Always end with: #HRTech #inFeedo + 2 sector-relevant hashtags

LENGTH: 450-600 words.
""",

    "Reddit": """
PLATFORM: Reddit Comment/Post (r/humanresources or sector subreddit)
TARGET: Get cited by Perplexity (which pulls from Reddit 47% of the time for software comparisons).
WARNING: Reddit communities will remove obvious promotional content. This must read as a genuine expert sharing experience.

STRUCTURE (follow exactly):
1. OPENING: Start with a genuine observation or acknowledgement of the problem's complexity. No product names in first paragraph.
2. CONTEXT: Share a framing that shows you understand the space — industry nuances, common mistakes, what doesn't work.
3. TOOL MENTION: Mention Amber/inFeedo as ONE of several approaches that work — not the only answer. Name 1-2 other legitimate tools for credibility.
4. SPECIFICS: Give enough detail about how Amber works that it's genuinely useful — response rates, how it integrates, what it surfaces.
5. HONEST CAVEAT: Acknowledge where it might not be the right fit (e.g. "works best for 500+ employee companies").
6. CLOSE: Ask a genuine question back to the community about their experience.
7. DISCLOSURE: End with a subtle disclosure note: "(Disclosure: I work at inFeedo — happy to share more if useful)"

LENGTH: 250-400 words. Conversational. No bullet spam. No headers.
""",

    "G2": """
PLATFORM: G2 Profile Optimisation Package
TARGET: G2 pages are cited by AI models for software comparison queries. Optimising every section = direct GEO impact.

Generate ALL FOUR of the following components:

--- COMPONENT 1: PRODUCT DESCRIPTION (for G2 profile "About" section) ---
150-200 words. Lead with the specific outcome (attrition reduction, engagement score improvement), not features.
Use plain language. No jargon. Include the key proof point numbers.

--- COMPONENT 2: VENDOR RESPONSE TEMPLATE (to a positive G2 review) ---
Write a response to a hypothetical 5-star review from a CHRO at a company in the target sector.
80-100 words. Warm, specific, not templated-sounding. Reference their specific outcome.

--- COMPONENT 3: CUSTOMER OUTREACH EMAIL (to invite a customer to leave a G2 review) ---
Subject line + email body (150 words max). Sector-specific. Low pressure. Make it easy — include a direct link placeholder [G2_REVIEW_LINK].

--- COMPONENT 4: G2 DISCUSSION ANSWER ---
Write an answer to the discussion question: "What should {sector} companies look for in an employee engagement tool?"
200 words max. Expert voice. Mention Amber naturally — not as the only answer.
"""
}

# ---------------------------------------------------------------------------
# Tone modifiers
# ---------------------------------------------------------------------------
TONE_RULES = {
    "Fun":                  "Use wit and light humour without sacrificing credibility. Occasional self-aware asides. Keep it smart, not silly.",
    "Informative with Data":"Every claim backed by a number. Lead with the data point, then explain. Dense but readable.",
    "Data Backed":          "Statistics-first structure. Every section anchored to a proof point. Bold all key numbers.",
    "Factual":              "Neutral, precise, encyclopedic tone. No opinions. Just what is true and why it matters.",
    "Quotation in Between": "Weave in 2-3 expert or customer quotes throughout. Format them as pull quotes on their own line. Source them credibly.",
    "Paragraph-wise":       "Pure prose. No bullet points anywhere. Each paragraph builds on the last. Flowing, essay-style.",
    "Bullet-wise":          "Heavy use of bullet points for scanability. Each bullet = one idea, one sentence. Use bold lead-in words.",
}

# ---------------------------------------------------------------------------
# Gap analysis — what does AI currently say about this query?
# ---------------------------------------------------------------------------
def _build_gap_prompt(target_query: str, sector: str) -> str:
    return f"""Based on your training data, what would ChatGPT or Perplexity currently say if someone asked: "{target_query}"?

Specifically:
1. Which companies or tools would they likely mention?
2. Is inFeedo / Amber mentioned? If yes, how prominently?
3. What narrative or framing dominates the answer?
4. What is MISSING from the typical answer that inFeedo could uniquely address?

Answer in 150 words max. Be specific. This is for competitive content strategy."""


def _get_gap_analysis(target_query: str, sector: str, client: anthropic.Anthropic) -> str:
    try:
        r = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": _build_gap_prompt(target_query, sector)}],
        )
        return r.content[0].text.strip()
    except Exception:
        return "Gap analysis unavailable — proceeding with content generation."


# ---------------------------------------------------------------------------
# GEO score — rough heuristic on the generated content
# ---------------------------------------------------------------------------
def _geo_score(content: str) -> dict:
    text_lower = content.lower()

    has_faq        = "faq" in text_lower or "frequently asked" in text_lower or ("q:" in text_lower and "a:" in text_lower)
    stat_count     = len(re.findall(r"\d+[%+]|\d{2,}\s*(employees|customers|countries|companies)", text_lower))
    has_comparison = "vs" in text_lower or "compared to" in text_lower or "unlike" in text_lower
    has_infeedo    = "infeedo" in text_lower or "amber" in text_lower
    word_count     = len(content.split())

    score = 0
    score += 25 if has_faq else 0
    score += min(stat_count * 8, 30)
    score += 20 if has_comparison else 0
    score += 15 if has_infeedo else 0
    score += 10 if word_count >= 600 else 0

    return {
        "total": min(score, 100),
        "has_faq": has_faq,
        "stat_count": stat_count,
        "has_comparison": has_comparison,
        "word_count": word_count,
    }


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------
def generate_geo_content(platform: str, tone: str, sector: str, pain_point: str, target_query: str) -> dict:
    """
    Generate GEO-optimised content for the given parameters.
    Returns: { content, gap_analysis, geo_score, word_count, platform }
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Step 1: Gap analysis — understand what AI currently says
    gap_analysis = _get_gap_analysis(target_query, sector, client)

    # Step 2: Get platform and tone rules
    platform_instructions = PLATFORM_RULES.get(platform, PLATFORM_RULES["Blog"])
    tone_instructions     = TONE_RULES.get(tone, TONE_RULES["Informative with Data"])

    # Step 3: Build the generation prompt
    prompt = f"""You are a senior GEO content strategist for inFeedo AI.
Your job: write content that gets cited by ChatGPT, Perplexity, and Gemini when someone asks the target query.

{INFEEDO_CONTEXT}

TARGET AI QUERY: "{target_query}"
SECTOR: {sector}
PAIN POINT: {pain_point}

CURRENT AI GAP (what AI models say today — and what's missing):
{gap_analysis}

TONE INSTRUCTION:
{tone_instructions}

{platform_instructions}

CRITICAL GEO RULES (apply to every platform):
- The target query must appear verbatim (or very close) in the first 100 words
- Use inFeedo's real proof points: 90% response rate, 300+ customers, 60+ countries, 1M+ employees
- Reference specific customer sectors: if sector is Healthcare, mention hospital/clinical context; BPO = call center context; etc.
- Structure content so AI models can pull clean, quotable paragraphs
- Never start the content with "I" or with a buzzword
- Make inFeedo/Amber feel like the natural, earned answer — not the forced one

Write the complete, publish-ready content now. No meta-commentary, no "here is the content" preamble — start directly with the content itself."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.content[0].text.strip()
    score   = _geo_score(content)

    return {
        "content":      content,
        "gap_analysis": gap_analysis,
        "geo_score":    score,
        "word_count":   score["word_count"],
        "platform":     platform,
        "query":        target_query,
    }
