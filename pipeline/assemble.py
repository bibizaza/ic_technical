"""
Stage: assemble

Reads complete draft_state.json (with subtitles filled by Claude Code)
and builds the final PowerPoint presentation.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from pptx import Presentation
from pptx.util import Cm, Inches, Pt
import copy

log = logging.getLogger(__name__)


def _load_draft(draft_path: str) -> dict:
    with open(draft_path) as f:
        return json.load(f)


def _find_placeholder(slide, ph_name: str):
    """Find a placeholder by name (case-insensitive partial match)."""
    for shape in slide.shapes:
        if ph_name.lower() in shape.name.lower():
            return shape
    return None


def _set_text(slide, ph_name: str, text: str) -> bool:
    """Set text in a placeholder. Returns True if found."""
    shape = _find_placeholder(slide, ph_name)
    if shape and shape.has_text_frame:
        shape.text_frame.paragraphs[0].runs[0].text = text if shape.text_frame.paragraphs[0].runs else text
        try:
            shape.text_frame.paragraphs[0].runs[0].text = text
        except IndexError:
            shape.text_frame.paragraphs[0].text = text
        return True
    return False


def _insert_image(slide, img_path: str, left_cm: float, top_cm: float, width_cm: float, height_cm: float) -> bool:
    """Insert an image at specified position. Returns True if successful."""
    if not img_path or not Path(img_path).exists():
        return False
    try:
        slide.shapes.add_picture(
            img_path,
            Cm(left_cm), Cm(top_cm), Cm(width_cm), Cm(height_cm)
        )
        return True
    except Exception as e:
        log.warning("Failed to insert image %s: %s", img_path, e)
        return False


def run_assemble(
    draft_path: str = "draft_state.json",
    template_path: Optional[str] = None,
    output_path: Optional[str] = None,
    history_path: str = "market_compass/data/history.json",
    config_path: str = "config/tickers.yaml",
) -> str:
    """
    Build the final PowerPoint from draft_state.json.

    Returns: path to the output PPTX file.
    """
    # Load draft
    draft = _load_draft(draft_path)
    ic_date = draft["date"]
    instruments = draft["instruments"]
    breadth = draft.get("breadth", [])
    fundamentals = draft.get("fundamentals", {})

    log.info("Assembling presentation for date: %s", ic_date)

    # Validate subtitles
    missing_subtitles = [nm for nm, d in instruments.items() if not d.get("subtitle")]
    if missing_subtitles:
        log.warning(
            "The following instruments have no subtitle: %s",
            ", ".join(missing_subtitles)
        )

    # Resolve template
    if template_path is None:
        dropbox_path = os.environ.get(
            "IC_DROPBOX_PATH",
            "/Users/larazanella/Library/CloudStorage/Dropbox/Tools_In_Construction/ic",
        )
        template_path = str(Path(dropbox_path) / "template.pptx")

    if not Path(template_path).exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    # Resolve output path
    if output_path is None:
        dropbox_path = os.environ.get(
            "IC_DROPBOX_PATH",
            "/Users/larazanella/Library/CloudStorage/Dropbox/Tools_In_Construction/ic",
        )
        date_str = ic_date.replace("-", "")
        output_path = str(Path(dropbox_path) / f"Market_Compass_{date_str}.pptx")

    # Load template
    prs = Presentation(template_path)
    log.info("Loaded template: %s (%d slides)", template_path, len(prs.slides))

    # Build presentation using existing technical_analysis modules
    _build_slides(prs, draft, ic_date, instruments, breadth, fundamentals)

    # Save
    prs.save(output_path)
    log.info("Saved presentation: %s", output_path)

    # Update history.json
    _update_history(history_path, ic_date, instruments)

    print(f"\n✓ Presentation assembled: {output_path}")
    print(f"  Date: {ic_date}")
    print(f"  Instruments: {len(instruments)}")
    return output_path


def _build_slides(prs, draft, ic_date, instruments, breadth, fundamentals):
    """
    Build all slides using the existing technical_analysis modules.
    Each instrument has pre-rendered chart_path from the prepare stage.
    """
    import importlib
    from pipeline.momentum import INSTRUMENT_MAP

    # Module mapping
    chart_module_map = {
        "S&P 500":    ("technical_analysis.equity.spx",       "spx"),
        "CSI 300":    ("technical_analysis.equity.csi",       "csi"),
        "Nikkei 225": ("technical_analysis.equity.nikkei",    "nikkei"),
        "TASI":       ("technical_analysis.equity.tasi",      "tasi"),
        "Sensex":     ("technical_analysis.equity.sensex",    "sensex"),
        "DAX":        ("technical_analysis.equity.dax",       "dax"),
        "SMI":        ("technical_analysis.equity.smi",       "smi"),
        "IBOV":       ("technical_analysis.equity.ibov",      "ibov"),
        "MEXBOL":     ("technical_analysis.equity.mexbol",    "mexbol"),
        "Gold":       ("technical_analysis.commodity.gold",   "gold"),
        "Silver":     ("technical_analysis.commodity.silver", "silver"),
        "Platinum":   ("technical_analysis.commodity.platinum","platinum"),
        "Palladium":  ("technical_analysis.commodity.palladium","palladium"),
        "Oil":        ("technical_analysis.commodity.oil",    "oil"),
        "Copper":     ("technical_analysis.commodity.copper", "copper"),
        "Bitcoin":    ("technical_analysis.crypto.bitcoin",   "bitcoin"),
        "Ethereum":   ("technical_analysis.crypto.ethereum",  "ethereum"),
        "Ripple":     ("technical_analysis.crypto.ripple",    "ripple"),
        "Solana":     ("technical_analysis.crypto.solana",    "solana"),
        "Binance":    ("technical_analysis.crypto.binance",   "binance"),
    }

    used_date = ic_date

    for name, data in instruments.items():
        if name not in chart_module_map:
            continue

        mod_path, prefix = chart_module_map[name]

        try:
            mod = importlib.import_module(mod_path)

            # Find the slide for this instrument
            # In production PPTX, slides are pre-created in template
            # We look for the slide matching this instrument by placeholder content
            slide = _find_instrument_slide(prs, name, prefix)
            if slide is None:
                log.warning("Could not find slide for %s — skipping", name)
                continue

            # Insert technical score
            insert_tech_fn = getattr(mod, f"insert_{prefix}_technical_score_number", None)
            if insert_tech_fn:
                insert_tech_fn(prs, slide)  # Note: these functions may need the full prs

            # Insert subtitle
            subtitle = data.get("subtitle") or ""
            insert_sub_fn = getattr(mod, f"insert_{prefix}_subtitle", None)
            if insert_sub_fn:
                insert_sub_fn(prs, subtitle)

            # Insert chart image (from pre-rendered PNG)
            chart_path = data.get("chart_path", "")
            if chart_path and Path(chart_path).exists():
                insert_chart_fn = getattr(mod, f"insert_{prefix}_technical_chart", None)
                # For pipeline v2: directly place the image into the slide
                # (bypassing the excel-dependent insert function)
                _replace_chart_in_slide(slide, chart_path, prefix)

        except Exception as e:
            log.warning("Failed to build slide for %s: %s", name, e)


def _find_instrument_slide(prs, instrument_name: str, prefix: str):
    """Find the slide corresponding to an instrument by searching slide titles."""
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                text = shape.text_frame.text.lower()
                if prefix.lower() in text or instrument_name.lower() in text:
                    return slide
    return None


def _replace_chart_in_slide(slide, chart_path: str, prefix: str):
    """Replace the chart placeholder in a slide with the pre-rendered PNG."""
    for shape in slide.shapes:
        if "chart" in shape.name.lower() or "technical" in shape.name.lower():
            left = shape.left
            top = shape.top
            width = shape.width
            height = shape.height
            # Remove old shape (if picture placeholder)
            sp = shape._element
            sp.getparent().remove(sp)
            # Add new picture
            try:
                slide.shapes.add_picture(chart_path, left, top, width, height)
            except Exception as e:
                log.warning("Could not replace chart in slide for %s: %s", prefix, e)
            return


def _update_history(history_path: str, ic_date: str, instruments: dict) -> None:
    """Update history.json with current week's scores."""
    path = Path(history_path)
    if path.exists():
        with open(path) as f:
            history = json.load(f)
    else:
        history = {}

    for name, data in instruments.items():
        if name not in history:
            history[name] = []

        entry = {
            "date":               ic_date,
            "dmas":               data.get("dmas"),
            "technical_score":    data.get("technical"),
            "momentum_score":     data.get("momentum"),
            "price_vs_50ma_pct":  _parse_pct(data.get("vs_50d", "0%")),
            "price_vs_100ma_pct": _parse_pct(data.get("vs_100d", "0%")),
            "price_vs_200ma_pct": _parse_pct(data.get("vs_200d", "0%")),
            "rating":             data.get("rating"),
        }

        # Remove duplicate entries for the same date
        history[name] = [h for h in history[name] if h.get("date") != ic_date]
        history[name].append(entry)
        # Keep last 52 weeks
        history[name] = sorted(history[name], key=lambda x: x["date"])[-52:]

    with open(path, "w") as f:
        json.dump(history, f, indent=2)

    log.info("Updated history.json (%d instruments)", len(instruments))


def _parse_pct(pct_str: str) -> float:
    """Parse '+1.5%' or '-2.3%' to float."""
    try:
        return float(pct_str.replace("%", "").replace("+", ""))
    except (ValueError, AttributeError):
        return 0.0
