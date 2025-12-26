# Market Compass Subtitle Generator

Automatically generates chart commentary for Market Compass weekly reports based on DMAS scores, technical indicators, and moving average positions.

## Features

- **Automatic Subtitle Generation** - Eliminates manual commentary writing
- **Priority-Based Decision Tree** - Routes to most relevant pattern (MA crosses → Dramatic changes → DMAS levels)
- **Anti-Repetition Logic** - Tracks last pattern used per asset to ensure variety
- **60+ Pattern Templates** - Organized by category (bullish, neutral, bearish, special events)
- **Batch Processing** - Generate subtitles for entire weekly report in one call
- **Configurable Length** - Automatic truncation with sentence boundary awareness

## Quick Start

### Single Asset

```python
from market_compass.subtitle_generator import generate_subtitle

asset_data = {
    "asset_name": "S&P 500",
    "dmas": 75,
    "technical_score": 70,
    "momentum_score": 80,
    "price_vs_50ma": "above",
    "price_vs_100ma": "above",
    "price_vs_200ma": "above",
    "dmas_prev_week": 72,
    "ma_cross_event": None,
    "channel_color": "green",
    "near_support": False,
    "near_resistance": False,
    "at_ath": False,
    "price_target": None
}

result = generate_subtitle(asset_data)
print(result["subtitle"])
# Output: "The picture remains bullish with strong momentum"
```

### Batch Processing

```python
from market_compass.subtitle_generator import SubtitleGenerator

generator = SubtitleGenerator()

# Process multiple assets
assets = [asset_data_1, asset_data_2, asset_data_3]
results = generator.generate_batch(assets)

for result in results:
    print(f"{result['asset_name']}: {result['subtitle']}")
```

## Input Schema

Each asset requires the following data:

```python
{
    "asset_name": str,           # e.g., "S&P 500", "Gold", "Bitcoin"
    "asset_class": str,          # "equity" | "commodity" | "crypto"
    "dmas": int,                 # 0-100, equally-weighted avg of tech + mom
    "technical_score": int,      # 0-100
    "momentum_score": int,       # 0-100
    "rating": str,               # Auto-calculated if not provided
    "price_vs_50ma": str,        # "above" | "below" | "at"
    "price_vs_100ma": str,       # "above" | "below" | "at"
    "price_vs_200ma": str,       # "above" | "below" | "at"
    "dmas_prev_week": int,       # Previous week DMAS for WoW change
    "rating_prev_week": str,     # Previous week rating
    "ma_cross_event": str|None,  # See MA Cross Events below
    "channel_color": str,        # "green" | "red" (regression channel)
    "near_support": bool,        # Price within 2% of support
    "near_resistance": bool,     # Price within 2% of resistance
    "at_ath": bool,              # At or near all-time high
    "price_target": float|None   # Optional price target
}
```

### MA Cross Events

- `"golden_cross"` - 50d MA crosses above 200d MA
- `"death_cross"` - 50d MA crosses below 200d MA
- `"crossed_above_50"` - Price crosses above 50d MA
- `"crossed_below_50"` - Price crosses below 50d MA
- `"crossed_above_100"` - Price crosses above 100d MA
- `"crossed_below_100"` - Price crosses below 100d MA
- `"crossed_above_200"` - Price crosses above 200d MA
- `"crossed_below_200"` - Price crosses below 200d MA
- `None` - No cross event

## Output Schema

```python
{
    "subtitle": str,        # Generated commentary (60-120 characters)
    "pattern_used": str,    # Pattern category for debugging
    "rating": str,          # Rating label based on DMAS
    "truncated": bool       # Whether subtitle was truncated
}
```

## Decision Logic

Subtitles are generated using a priority-based decision tree:

### Priority 1: MA Cross Events (Highest Priority)
Most newsworthy events take precedence. Examples:
- Death cross: "A short-lived rebound accompanied by a death cross"
- Golden cross (strong): "Golden cross confirms the bullish setup"
- Golden cross (weak): "S&P 500 has been stuck in the current range for the past weeks, despite a golden cross"
- Crossed above MA: "Successful rebound on the 50d MA"
- Crossed below MA: "Dangerous setup for S&P 500 that just closed below the 50MA"

### Priority 2: Dramatic WoW Changes (|Δ| > 15)
Large momentum shifts are highlighted:
- DMAS +20 WoW: "A surge in momentum lifts DMAS and moves S&P 500 closer to bullish territory"
- DMAS -20 WoW: "A dramatic picture for S&P 500 that crossed all its MA in a week"

### Priority 3: DMAS Level Routing

#### Bullish (DMAS > 65)
Routes based on component scores:
- **Strong bullish** (Tech ≥60, Mom ≥80): "The picture remains bullish with strong momentum"
- **Bullish at ATH**: "New all-time highs are supported by strong technical and momentum"
- **Bullish with target**: "The bull trend is intact, S&P 500 could go to 4800 in the coming weeks"
- **Bullish caution** (Tech <55, Mom ≥80): "While the momentum is still high, the technical score has weakened, calling for some caution"
- **Bullish correction** (red channel or negative WoW): "The picture remains bullish despite the current correction"

#### Neutral (DMAS 45-65)
Focuses on component divergence:
- **High tech, low momentum** (Tech ≥60, Mom <40): "The technical score is still supporting the bull movement despite the poor momentum"
- **Low tech, high momentum** (Tech <50, Mom ≥70): "High momentum score may offset the current technical weakness"
- **Consolidation** (|WoW| <5): "The technical score has been stable over the past weeks. Ongoing consolidation"
- **Turning point** (near support/resistance): "S&P 500 is at a turning point: technical suggest uptrend but momentum calls caution"

#### Bearish (DMAS < 45)
Highlights weakness:
- **Deep bearish** (DMAS <25): "S&P 500 is deeply engulfed in negative territory"
- **No hope** (Tech <40, Mom <40): "No short term hopes of a sustained rebound with challenging technical and momentum"
- **Stuck below MA**: "Still below the 50-day MA that seems to act as a strong resistance"
- **Silver lining** (Tech ≥50 OR Mom ≥50, above 200d): "Very weak picture but S&P 500 has so far managed to stay above the 200d MA"
- **Deteriorating** (WoW <-5): "The technical setup is weakening even more"

## Rating Thresholds

DMAS scores map to ratings:

| DMAS Range | Rating |
|------------|--------|
| 0-20 | Strongly Bearish |
| 21-35 | Bearish |
| 36-45 | Slightly Bearish |
| 46-55 | Neutral |
| 56-65 | Slightly Bullish |
| 66-80 | Bullish |
| 81-100 | Strongly Bullish |

## Pattern Categories

The module includes 60+ templates across 20+ categories:

**Bullish Categories:**
- `bullish_strong` - Strong momentum and technical (6 variations)
- `bullish_ath` - At all-time high (3 variations)
- `bullish_target` - Near price target (3 variations)
- `bullish_caution` - High momentum, weak technical (3 variations)
- `bullish_correction` - Bullish despite pullback (4 variations)

**Neutral Categories:**
- `neutral_tech_offset` - High tech offsets low momentum (3 variations)
- `neutral_mom_offset` - High momentum offsets low tech (3 variations)
- `neutral_consolidation` - Sideways movement (3 variations)
- `neutral_turning` - At inflection point (3 variations)
- `neutral_default` - Generic neutral (3 variations)

**Bearish Categories:**
- `bearish_deep` - Deeply negative (3 variations)
- `bearish_no_hope` - Both components weak (3 variations)
- `bearish_stuck` - Below MA resistance (3 variations)
- `bearish_silver` - Weak but not collapsed (3 variations)
- `bearish_deteriorating` - Getting worse (3 variations)
- `bearish_default` - Generic bearish (3 variations)

**Special Event Categories:**
- `ma_cross_up` - Upward MA crosses (3 variations)
- `ma_cross_down` - Downward MA crosses (3 variations)
- `golden_cross` - Golden cross (3 variations)
- `golden_cross_weak` - Failed golden cross (3 variations)
- `death_cross` - Death cross (3 variations)
- `dramatic_surge` - Large WoW improvement (3 variations)
- `dramatic_collapse` - Large WoW deterioration (3 variations)
- `extreme_momentum_high` - Momentum = 100 (3 variations)
- `extreme_momentum_low` - Momentum = 0 (3 variations)

## Anti-Repetition Logic

The `SubtitleGenerator` class maintains state across weeks:

```python
generator = SubtitleGenerator()

# Week 1
result1 = generator.generate(spx_data)
# Output: "The picture remains bullish with strong momentum"

# Week 2 (same conditions)
result2 = generator.generate(spx_data)
# Output: "Momentum remains strong as well as technical. The setup is clean and bullish"

# Week 3 (same conditions)
result3 = generator.generate(spx_data)
# Output: "S&P 500 has all the technical elements to go higher in the coming weeks"
```

The generator filters out the last used pattern for each asset, ensuring variety even when conditions remain similar.

## PowerPoint Integration

```python
from pptx import Presentation
from market_compass.subtitle_generator import SubtitleGenerator

def generate_market_compass_ppt(df, output_path):
    """Generate Market Compass with automatic subtitles."""
    generator = SubtitleGenerator()
    prs = Presentation("market_compass_template.pptx")

    for idx, row in df.iterrows():
        # Prepare asset data
        asset_data = {
            "asset_name": row["asset_name"],
            "dmas": int(row["dmas"]),
            "technical_score": int(row["technical_score"]),
            "momentum_score": int(row["momentum_score"]),
            # ... other fields ...
        }

        # Generate subtitle
        result = generator.generate(asset_data, max_length=120)

        # Insert into slide
        slide = prs.slides[idx + 1]
        for shape in slide.shapes:
            if shape.name.lower() == "subtitle":
                shape.text = result["subtitle"]
                break

    prs.save(output_path)
```

## Testing

Run the comprehensive test suite:

```bash
python market_compass/subtitle_generator/tests/test_generator.py
```

The test suite includes:
- ✓ 7 rating threshold tests
- ✓ 13+ scenario tests (strong bullish, neutral, bearish, MA crosses, etc.)
- ✓ Anti-repetition tests
- ✓ Batch generation tests
- ✓ Edge case tests (extreme values, truncation, missing fields)
- ✓ 30 total tests

## Project Structure

```
market_compass/subtitle_generator/
├── __init__.py              # Package exports
├── patterns.py              # 60+ pattern templates, rating function
├── decision_tree.py         # Priority routing logic
├── generator.py             # Main SubtitleGenerator class
├── tests/
│   └── test_generator.py    # Comprehensive test suite
├── example_integration.py   # Integration examples
└── README.md               # This file
```

## API Reference

### `SubtitleGenerator`

Main class for subtitle generation with anti-repetition.

**Methods:**
- `generate(asset_data, last_week_data=None, max_length=120)` → dict
- `generate_batch(assets_data, max_length=120)` → list[dict]
- `reset_history()` → None
- `get_stats()` → dict

### `generate_subtitle()`

Convenience function for single-shot usage.

```python
generate_subtitle(
    asset_data: dict,
    last_week_data: dict = None,
    last_pattern: str = None,
    max_length: int = 120
) -> dict
```

### `get_rating()`

Convert DMAS score to rating label.

```python
get_rating(dmas: int) -> str
```

## Edge Cases

The module handles:

1. **Extreme momentum** (0 or 100): Always mentioned explicitly
2. **All MAs breached in one week**: "Dramatic picture for S&P 500 that crossed all its MA in a week"
3. **Rating changed but DMAS similar**: Focuses on component shifts
4. **Price target reached**: Celebrates achievement
5. **Missing optional fields**: Uses sensible defaults
6. **Truncation**: Preserves sentence boundaries when possible

## Examples

See `example_integration.py` for:
- Single asset generation
- Weekly report batch processing
- PowerPoint integration template
- Multi-week consistency demo
- DataFrame integration

## Design Principles

1. **Newsworthy First** - MA crosses and dramatic changes take priority
2. **Context Matters** - Same DMAS can have different subtitles based on components
3. **Variety** - Anti-repetition ensures fresh commentary each week
4. **Accuracy** - 120-character limit keeps subtitles concise and readable
5. **Debugging** - `pattern_used` field helps understand routing decisions

## Future Enhancements

Potential additions:
- Rhetorical questions for turning points ("Dead cat bounce or start of a new trend?")
- Price range detection ("stuck in 4200-4400 range")
- Ordinal target tracking ("approaching 2nd target of 4800")
- Volume/breadth integration
- Sector rotation commentary
- Correlation analysis

## License

Internal use only - Herculis Partners

## Authors

Developed for Market Compass weekly report automation.
