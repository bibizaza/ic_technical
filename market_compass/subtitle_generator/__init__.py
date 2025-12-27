"""
Market Compass Subtitle Generator.

Hybrid system using:
1. Deterministic fact extraction
2. Claude API for unique subtitle generation

Also includes legacy pattern-based generator for fallback.
"""

# Fact extraction (deterministic)
from .fact_extractor import (
    extract_facts,
    format_facts_for_prompt,
    MarketFacts,
    Rating,
    get_rating,
)

# Claude API generation
from .claude_generator import (
    generate_subtitle,
    generate_batch,
    quick_generate,
)

# Style examples
from .style_examples import (
    STYLE_EXAMPLES,
    RATING_SPECIFIC_EXAMPLES,
    get_examples_for_rating,
)

# Legacy pattern-based generator (for fallback)
from .generator import (
    SubtitleGenerator,
    SubtitleTracker,
    generate_subtitle as generate_subtitle_legacy,
    validate_subtitle_language,
)
from .patterns import (
    PATTERNS,
    get_ma_dynamics,
    get_high_low_dynamics,
    add_context_if_needed,
)

__all__ = [
    # Fact extraction
    'extract_facts',
    'format_facts_for_prompt',
    'MarketFacts',
    'Rating',
    'get_rating',
    # Claude generation
    'generate_subtitle',
    'generate_batch',
    'quick_generate',
    # Style examples
    'STYLE_EXAMPLES',
    'RATING_SPECIFIC_EXAMPLES',
    'get_examples_for_rating',
    # Legacy (fallback)
    'SubtitleGenerator',
    'SubtitleTracker',
    'generate_subtitle_legacy',
    'validate_subtitle_language',
    'PATTERNS',
    'get_ma_dynamics',
    'get_high_low_dynamics',
    'add_context_if_needed',
]

__version__ = '4.0.0'
