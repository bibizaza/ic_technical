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


def strip_instrument_name(subtitle: str, instrument: str) -> str:
    """Remove instrument name if it appears at the start of a subtitle line."""
    if not subtitle or not instrument:
        return subtitle
    
    # Build variants: "S&P 500", "S&P 500's", "Nikkei 225's", etc.
    variants = [instrument, instrument.upper(), instrument.lower(), instrument.title()]
    # Add possessive forms
    variants += [f"{v}'s" for v in variants]
    # Add common short forms
    short_forms = {
        "Ibovespa": ["IBOV", "Ibov", "IBOV's", "Ibov's"],
        "IBOV": ["Ibovespa", "Ibovespa's"],
        "Dax": ["DAX", "DAX's", "Dax's"],
        "MEXBOL": ["Mexbol", "MEXBOL's", "Mexbol's"],
        "Nikkei 225": ["Nikkei", "Nikkei's"],
        "Sensex": ["BSE Sensex", "BSE Sensex's"],
        "Gold": ["Gold's"],
        "Silver": ["Silver's"],
        "Platinum": ["Platinum's"],
        "Palladium": ["Palladium's"],
        "Bitcoin": ["Bitcoin's", "BTC", "BTC's"],
        "Ethereum": ["Ethereum's", "ETH", "ETH's"],
        "Ripple": ["Ripple's", "XRP", "XRP's"],
        "Solana": ["Solana's", "SOL", "SOL's"],
        "Binance": ["Binance's", "BNB", "BNB's"],
        "Oil": ["Oil's", "WTI", "WTI's"],
        "Copper": ["Copper's"],
    }
    if instrument in short_forms:
        variants += short_forms[instrument]
    
    lines = subtitle.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        for v in sorted(variants, key=len, reverse=True):  # Longest first
            if stripped.startswith(v):
                stripped = stripped[len(v):].lstrip(" ',-:").strip()
                # Capitalize first letter
                if stripped:
                    stripped = stripped[0].upper() + stripped[1:]
                break
        cleaned_lines.append(stripped)
    
    return '\n'.join(cleaned_lines)


def detect_repeated_phrases(results: list, min_words: int = 3) -> list:
    """Detect repeated multi-word phrases across subtitles.
    
    Returns list of warnings: [{phrase, instruments}]
    """
    from collections import defaultdict
    
    phrase_map = defaultdict(list)  # phrase -> [instrument names]
    
    for r in results:
        subtitle = r.get("deepseek_subtitle", "")
        if subtitle.startswith('['):
            continue  # Skip errors
        
        # Normalize: lowercase, strip punctuation
        clean = re.sub(r'[^\w\s]', '', subtitle.lower())
        words = clean.split()
        
        # Extract all N-grams of length min_words to min_words+3
        seen_in_this = set()  # Avoid counting same phrase twice in one subtitle
        for n in range(min_words, min(min_words + 4, len(words) + 1)):
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i:i+n])
                if phrase not in seen_in_this:
                    seen_in_this.add(phrase)
                    phrase_map[phrase].append(r["instrument"])
    
    # Filter to phrases appearing in 2+ different instruments
    warnings = []
    seen_superstrings = set()
    for phrase, instruments in sorted(phrase_map.items(), key=lambda x: -len(x[0].split())):
        if len(instruments) >= 2 and len(set(instruments)) >= 2:
            # Skip if this is a substring of an already-reported longer phrase
            is_sub = False
            for seen in seen_superstrings:
                if phrase in seen:
                    is_sub = True
                    break
            if not is_sub:
                seen_superstrings.add(phrase)
                warnings.append({
                    "phrase": phrase,
                    "instruments": list(set(instruments)),
                    "count": len(set(instruments)),
                })
    
    # Sort by count descending, then phrase length descending
    warnings.sort(key=lambda x: (-x["count"], -len(x["phrase"].split())))
    return warnings[:15]  # Top 15 most problematic


def call_deepseek(system_prompt: str, user_prompt: str, instrument: str = "", previous_subtitles: list = None) -> str:
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

CRITICAL RULES:
- NEVER start with "[Rating] dynamics/setup continue/extend/persist/build" — find a unique angle
- BANNED WORDS: "dynamics", "setup", "pointing to", "suggesting", "divergence". Use plain English instead
- BANNED PHRASES: "14-week bullish run", "14-week bullish streak", "recovery hinges on", "recovery depends on", "exceptional technical strength remains", "outlook persists", "bearish outlook persists", "bullish streak remains unbroken", "maintains bullish trend", "resilience despite DMAS decline", "momentum strength key to", "hinges on reclaiming 50d MA", "momentum yet to confirm technical alignment", "remains trapped below all MAs", "as DMAS drops". These exact phrases and close variants have been overused. Find completely different constructions to express the same idea
- Line 1 should highlight the MOST RELEVANT FACT: what is the single most important thing about this asset RIGHT NOW? A key level being tested? A streak? A score collapse? A correction depth? A reversal?
- Line 2 should answer WHAT TO WATCH NEXT: a condition ("recovery depends on reclaiming 50d MA"), a comparison ("weakest among equity indices"), a risk ("further decline if momentum fails to recover"), or a timeline ("third consecutive week of deterioration"). Use specific numbers when available (e.g., "3rd week of decline", "50d MA at risk", "rallied 12% from lows"). Line 2 must NOT just restate Line 1 in different words
- IGNORE small changes: a DMAS move from 95 to 90 is noise. A move from 65 to 45 is a story. Only mention score changes if they are significant (>15 points)
- DIFFERENTIATE: instruments in the same asset class must NOT sound alike. What makes THIS one unique right now? Different momentum levels? Different MA positions? Different correction depths? Different streak lengths?
- Reference the asset name at least once, but never as the first words of a line
- NEVER end with generic filler like "pointing to potential gains", "suggesting further upside", "points to continued advance", "suggests further strength", "supports further gains". Instead, end with something SPECIFIC: a level being tested, a timeframe, a comparison ("strongest momentum among precious metals"), or a conditional ("recovery depends on reclaiming 50d MA")
- ONLY reference facts from the data provided. NEVER invent macro narratives like "demand strengthens", "supply constraints", "domestic demand drives", "safe-haven appeal", "industrial demand". Stick strictly to: scores, MAs, streaks, corrections, price levels, DMAS changes, and rating duration
- Write for a sophisticated investor, NOT a quant. Avoid jargon like "divergence", "technical vs momentum". Instead say what it MEANS: "momentum racing ahead of technicals" or "price strength not yet confirmed by structure". If two scores disagree, explain the implication, not the math
- DO NOT combine unrelated facts into a false causal chain. "14-week rally from -23% correction" implies the rally started after the correction — only combine facts if the causal link is clear from the data. If unsure, state facts separately
- Never begin a subtitle line with the instrument name. The instrument name is already displayed as the slide title — repeating it wastes words and adds no value
- Before finalizing each subtitle, review the previously generated subtitles listed below. Do not reuse the same sentence structure, opening phrase, or closing phrase. Each instrument must read as a distinct observation. If you find yourself writing a pattern already used (e.g., "Recovery hinges on...", "...despite correction from highs"), rewrite it with a completely different construction
- Do not state any number (streak duration, correction percentage, DMAS change, MA distance) unless it is explicitly present or directly calculable from the data provided. Do not infer or estimate statistics that are not in the input. Directional claims must align with the scores: if Technical >= 80, do not suggest technical weakness; if Momentum <= 30, do not suggest momentum strength. When in doubt, describe the setup qualitatively rather than fabricating a statistic

GOOD (real examples):
"S&P 500 neutral for sixth straight week with DMAS down -16
Key test ahead: 50d MA must hold or risk shift to cautious"

"Sensex trapped below all MAs as DMAS tumbles 17 points
Weakest equity index — no recovery signal despite oversold RSI at 23"

"Bitcoin languishing below all MAs for 11 consecutive weeks
Recovery requires reclaiming 50d MA, currently 11% above price"

"Palladium's 14-week bullish run holds despite -15 DMAS drop
Momentum racing ahead of technicals — watch for confirmation or fade"

"Silver rallies +7.6% this week, strongest among precious metals
14-week streak intact but -23% from 52w high limits near-term upside"

"CSI 300 testing 50d MA resistance after 14 weeks above
Fragile neutral — weak momentum (22) risks tipping to cautious"

BAD (generic — never write these):
"Bullish dynamics continue with exceptional setup pointing to gains"
"Bearish setup persists with weak momentum suggesting decline"
"Solid technical vs exceptional momentum divergence points to upside"
"Strong momentum supports further gains ahead"

""" + (_build_previous_subtitles_block(previous_subtitles) if previous_subtitles else "") + """Generate subtitle (2 lines, no period, no labels):"""
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

        # Remove leaked reasoning and quote artifacts
        subtitle = re.sub(r'This subtitle\b.*', '', subtitle, flags=re.IGNORECASE).strip()
        subtitle = re.sub(r"Here(?:'s| is)\b.*", '', subtitle, flags=re.IGNORECASE).strip()
        subtitle = re.sub(r'Note:.*', '', subtitle, flags=re.IGNORECASE).strip()
        subtitle = subtitle.replace('"', '')
        subtitle = re.sub(r'\.\s*\n', '\n', subtitle)

        # Remove generic endings and labels
        for generic in ["pointing to potential gains", "suggesting further upside", 
                        "suggests potential gains", "pointing to further gains",
                        "suggests likely continued advance", "points to potential further gains",
                        "suggest potential gains", "pointing to potential further gains",
                        "suggests continued strength ahead", "supports further gains",
                        "**Line 1:**", "**Line 2:**", "Line 1:", "Line 2:", "**"]:
            subtitle = subtitle.replace(generic, "").strip().rstrip(",").rstrip(".")

        # Remove orphan fragments from banned phrase stripping
        lines = subtitle.split('\n')
        lines = [l for l in lines if len(l.split()) >= 5]
        subtitle = '\n'.join(lines)
        subtitle = re.sub(r'\n{2,}', '\n', subtitle).strip()

        # Clean up
        subtitle = subtitle.strip('"\'').rstrip('.')

        # Strip instrument name from start of each line
        subtitle = strip_instrument_name(subtitle, instrument)

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


def _build_previous_subtitles_block(previous_subtitles: list) -> str:
    """Build context block of previously generated subtitles for dedup."""
    if not previous_subtitles:
        return ""
    lines = ["PREVIOUSLY GENERATED SUBTITLES IN THIS BATCH (DO NOT reuse their phrasing, structure, or closing phrases):\n"]
    for entry in previous_subtitles:
        lines.append(f"- {entry['instrument']}: {entry['subtitle']}")
    lines.append("\n")
    return "\n".join(lines)


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
    previous_subtitles = []  # Accumulate for dedup context
    for i, entry in enumerate(prompts):
        instrument = entry.get("instrument", "Unknown")
        system_prompt = entry.get("system", "")
        user_prompt = entry.get("user", "")
        rating = get_rating_from_prompt(user_prompt)

        print(f"\n[{i+1}/{len(prompts)}] {instrument} ({rating})")

        # Call DeepSeek with previous subtitles for dedup
        print(f"  Calling DeepSeek... ({len(previous_subtitles)} prior subtitles in context)")
        deepseek_subtitle = call_deepseek(system_prompt, user_prompt, instrument, previous_subtitles)
        print(f"  -> {deepseek_subtitle[:70]}")

        results.append({
            "instrument": instrument,
            "rating": rating,
            "deepseek_subtitle": deepseek_subtitle,
        })

        # Accumulate for next call's dedup context
        if not deepseek_subtitle.startswith('['):  # Skip errors
            previous_subtitles.append({
                "instrument": instrument,
                "subtitle": deepseek_subtitle,
            })

    # Detect repeated phrases across subtitles
    warnings = detect_repeated_phrases(results)
    if warnings:
        print("\n" + "=" * 60)
        print("REPETITION WARNINGS")
        print("=" * 60)
        for w in warnings:
            instruments_str = ", ".join(w["instruments"])
            print(f'  "{w["phrase"]}" -> {instruments_str}')
    else:
        print("\n  No repeated phrases detected.")

    # Generate HTML
    html_path = OUTPUT_DIR / "shadow_comparison.html"
    generate_html(results, html_path, warnings)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  Results: {len(results)} instruments")
    print(f"  Warnings: {len(warnings)} repeated phrases")
    print(f"  Output: {html_path}")


def generate_html(results, output_path, warnings=None):
    """Generate HTML comparison table with optional repetition warnings."""
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
        .warnings {
            margin-top: 24px;
            padding: 16px 20px;
            background: #FEF2F2;
            border: 1px solid #FECACA;
            border-radius: 8px;
        }
        .warnings h2 {
            color: #991B1B;
            font-size: 16px;
            margin: 0 0 12px 0;
        }
        .warning-item {
            font-size: 13px;
            color: #7F1D1D;
            padding: 4px 0;
            border-bottom: 1px solid #FEE2E2;
        }
        .warning-item:last-child { border-bottom: none; }
        .warning-phrase { 
            font-weight: 600;
            background: #FEE2E2;
            padding: 1px 4px;
            border-radius: 3px;
        }
        .warning-instruments { color: #B91C1C; font-style: italic; }
        .no-warnings {
            margin-top: 24px;
            padding: 12px 20px;
            background: #F0FDF4;
            border: 1px solid #BBF7D0;
            border-radius: 8px;
            color: #166534;
            font-size: 14px;
        }
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
"""

    # Add warnings section
    if warnings:
        html += """    <div class="warnings">
        <h2>Repetition Warnings (shared phrases across instruments)</h2>
"""
        for w in warnings:
            instruments_str = ", ".join(w["instruments"])
            html += f"""        <div class="warning-item">
            <span class="warning-phrase">{w["phrase"]}</span>
            &rarr; <span class="warning-instruments">{instruments_str}</span>
        </div>
"""
        html += """    </div>
"""
    else:
        html += """    <div class="no-warnings">No repeated phrases detected across subtitles.</div>
"""

    html += """</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nHTML saved to: {output_path}")


if __name__ == "__main__":
    main()