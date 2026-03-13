#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shadow Simple - Replay logged prompts to DeepSeek.

Reads prompts_log.json (written by production IC run) and sends each to
DeepSeek via Ollama, producing a side-by-side comparison.

Usage:
    1. Run production IC pipeline (writes to /tmp/prompts_log.json)
    2. python shadow_simple.py

Output:
    - shadow_comparison.html
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import requests

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PROMPTS_LOG = Path("/tmp/prompts_log.json")
OUTPUT_DIR = Path(__file__).parent

OLLAMA_URL = "http://localhost:11434/api/generate"
DEEPSEEK_MODEL = "deepseek-r1:32b"


# ==============================================================================
# HELPERS
# ==============================================================================

def sanitize_unicode(text: str) -> str:
    """Replace Unicode smart quotes with ASCII equivalents."""
    if text is None:
        return ""
    replacements = {
        '\u201c': '"', '\u201d': '"',
        '\u2018': "'", '\u2019': "'",
        '\u2013': '-', '\u2014': '--',
        '\u2026': '...',
    }
    for u, a in replacements.items():
        text = text.replace(u, a)
    return text


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from DeepSeek response."""
    # Remove everything between <think> and </think>
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()


def call_deepseek(system_prompt: str, user_prompt: str) -> str:
    """Call DeepSeek via Ollama."""
    try:
        clean_system = sanitize_unicode(system_prompt)
        clean_user = sanitize_unicode(user_prompt)
        full_prompt = f"{clean_system}\n\n{clean_user}"
        
# Override prompt for enhanced two-line subtitles
        full_prompt = full_prompt.replace(
            "MAXIMUM 12 WORDS - Must fit one line",
            "EXACTLY 2 LINES, max 12 words each (24 words total)"
        ).replace(
            "Generate subtitle (max 12 words, no period):",
            """Generate a 2-line subtitle (no period, no "Line 1/Line 2" labels).

CRITICAL RULES FOR QUALITY:
- NEVER start with "[Rating] dynamics continue/extend/persist/build" — find a unique angle
- Line 1 should highlight the MOST RELEVANT FACT: what is the single most important thing about this asset RIGHT NOW? A key level being tested? A divergence between technical and momentum? A streak? A reversal?
- Line 2 should add PREDICTIVE context: what does the data suggest happens next? Use specific numbers when available (e.g., "3rd week of decline", "50d MA at risk", "rallied 12% from lows")
- IGNORE small changes: a DMAS move from 95 to 90 is noise. A move from 65 to 45 is a story. Only mention score changes if they are significant (>15 points)
- DIFFERENTIATE: Gold, Silver, Platinum should NOT sound the same. What makes each unique right now? Different momentum levels? Different MA positions? Different correction depths?
- Use the asset name or a specific reference, not generic "dynamics"
- NEVER end with generic conclusions like "pointing to potential gains", "suggesting further upside", "points to continued advance". Instead, end with something SPECIFIC: a level being tested, a timeframe, a divergence, a comparison ("strongest momentum among precious metals"), or a conditional ("recovery depends on reclaiming 50d MA")
- ONLY reference facts from the data provided. NEVER invent macro narratives like "demand strengthens", "supply constraints", "domestic demand drives", "safe-haven appeal", "industrial demand". Stick strictly to: scores, MAs, streaks, corrections, price levels, DMAS changes, and rating duration
- DO NOT combine unrelated facts into a false narrative. "14-week rally from -23% correction" implies the rally started after the correction — only combine facts if the causal link is clear from the data. If unsure, state facts separately

GOOD examples:
"Gold tests all-time highs with momentum accelerating above 90
Safe-haven bid strengthens as breadth rank holds at #1"

"S&P 500 stalls at 50d MA as momentum fades from January peaks  
Constructive but fragile — third consecutive week of narrowing breadth"

"Solana trapped 25% below 200d MA with no recovery signal
Weakest crypto setup as technical and momentum both deteriorate"

BAD examples (too generic):
"Bullish dynamics continue with exceptional setup pointing to gains"
"Bearish dynamics persist with weak momentum"

Generate subtitle (2 lines, no period, no labels):"""
        )

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": DEEPSEEK_MODEL,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "num_predict": 4096,
                    "temperature": 0.7,
                }
            },
            timeout=600,
            headers={"Content-Type": "application/json; charset=utf-8"}
        )

        if response.status_code != 200:
            return f"[HTTP {response.status_code}]"

        result = response.json()
        subtitle = result.get("response", "").strip()

        # Strip thinking blocks
        subtitle = strip_think_blocks(subtitle)

	# Remove generic endings and labels
        for generic in ["pointing to potential gains", "suggesting further upside", 
                        "suggests potential gains", "pointing to further gains",
                        "suggests likely continued advance", "points to potential further gains",
                        "suggest potential gains", "pointing to potential further gains",
                        "suggests continued strength ahead", "supports further gains",
                        "**Line 1:**", "**Line 2:**", "Line 1:", "Line 2:", "**"]:
            subtitle = subtitle.replace(generic, "").strip().rstrip(",").rstrip(".")

        # Clean up
        subtitle = subtitle.strip('"\'').rstrip('.')

        # Truncate to ~12 words
        words = subtitle.split()
        if len(words) > 30:
            subtitle = ' '.join(words[:28])

        return subtitle

    except requests.exceptions.ConnectionError:
        return "[unavailable - Ollama not running]"
    except requests.exceptions.Timeout:
        return "[timeout]"
    except Exception as e:
        return f"[error: {str(e)[:80]}]"


def get_rating_from_prompt(user_prompt: str) -> str:
    """Extract rating from prompt text."""
    if "Rating: Bullish" in user_prompt:
        return "Bullish"
    elif "Rating: Constructive" in user_prompt:
        return "Constructive"
    elif "Rating: Neutral" in user_prompt:
        return "Neutral"
    elif "Rating: Cautious" in user_prompt:
        return "Cautious"
    elif "Rating: Bearish" in user_prompt:
        return "Bearish"
    return "Unknown"


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print("SHADOW SIMPLE - Replay prompts to DeepSeek")
    print("=" * 60)

    # Check prompts log exists
    if not PROMPTS_LOG.exists():
        print(f"\nError: {PROMPTS_LOG} not found")
        print("Run the production IC pipeline first to generate prompts.")
        sys.exit(1)

    # Read prompts (one JSON per line)
    print(f"\nReading prompts from: {PROMPTS_LOG}")
    prompts = []
    with open(PROMPTS_LOG, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    prompts.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    print(f"Found {len(prompts)} prompts")

    if not prompts:
        print("No prompts to process.")
        sys.exit(1)

    # Process each prompt
    print("\nCalling DeepSeek for each prompt...")
    print("-" * 60)

    results = []
    for i, entry in enumerate(prompts):
        instrument = entry.get("instrument", "Unknown")
        system_prompt = entry.get("system", "")
        user_prompt = entry.get("user", "")
        rating = get_rating_from_prompt(user_prompt)

        print(f"\n[{i+1}/{len(prompts)}] {instrument} ({rating})")

        # Call DeepSeek
        print(f"  Calling DeepSeek...")
        deepseek_subtitle = call_deepseek(system_prompt, user_prompt)
        print(f"  -> {deepseek_subtitle[:70]}")

        results.append({
            "instrument": instrument,
            "rating": rating,
            "deepseek_subtitle": deepseek_subtitle,
        })

    # Generate HTML
    html_path = OUTPUT_DIR / "shadow_comparison.html"
    generate_html(results, html_path)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  Results: {len(results)} instruments")
    print(f"  Output: {html_path}")


def generate_html(results, output_path):
    """Generate HTML comparison table."""
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Shadow Comparison - DeepSeek Subtitles</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }
        h1 { color: #1a1a2e; border-bottom: 2px solid #4a4e69; padding-bottom: 10px; }
        .timestamp { color: #666; font-size: 14px; margin-bottom: 20px; }
        table {
            border-collapse: collapse;
            width: 100%;
            background: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }
        th { background: #1a1a2e; color: white; padding: 12px 16px; text-align: left; }
        td { padding: 12px 16px; border-bottom: 1px solid #eee; }
        tr:hover { background: #f8f9fa; }
        .rating-bullish { color: #22c55e; font-weight: 600; }
        .rating-constructive { color: #84cc16; font-weight: 600; }
        .rating-neutral { color: #eab308; font-weight: 600; }
        .rating-cautious { color: #f97316; font-weight: 600; }
        .rating-bearish { color: #ef4444; font-weight: 600; }
        .subtitle { font-style: italic; color: #333; }
        .error { color: #ef4444; font-style: italic; }
    </style>
</head>
<body>
    <h1>Shadow Comparison - DeepSeek Subtitles</h1>
    <p class="timestamp">Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f"""<br>
    Model: {DEEPSEEK_MODEL}</p>
    <table>
        <thead>
            <tr>
                <th>Instrument</th>
                <th>Rating</th>
                <th>DeepSeek Subtitle</th>
            </tr>
        </thead>
        <tbody>
"""

    for r in results:
        rating_class = f"rating-{r['rating'].lower()}"
        sub_class = "error" if r['deepseek_subtitle'].startswith('[') else "subtitle"
        html += f"""            <tr>
                <td><strong>{r['instrument']}</strong></td>
                <td class="{rating_class}">{r['rating']}</td>
                <td class="{sub_class}">{r['deepseek_subtitle']}</td>
            </tr>
"""

    html += """        </tbody>
    </table>
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nHTML saved to: {output_path}")


if __name__ == "__main__":
    main()
