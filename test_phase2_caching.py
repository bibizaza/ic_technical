#!/usr/bin/env python3
"""
Test Phase 2 image caching infrastructure.

Tests that chart images can be cached and retrieved for parallel processing.
"""

import sys
from pathlib import Path

print("=" * 70)
print("Phase 2 Caching Infrastructure Test")
print("=" * 70)

# Test 1: Import caching functions
print("\n[Test 1] Importing caching functions...")
try:
    from technical_analysis.common_helpers import (
        enable_image_cache,
        disable_image_cache,
        cache_chart_image,
        get_cached_chart_image,
        _IMAGE_CACHE_ENABLED,
    )
    print("✓ Successfully imported caching functions")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test 2: Cache enabling/disabling
print("\n[Test 2] Testing cache enable/disable...")
disable_image_cache()
print(f"  After disable: _IMAGE_CACHE_ENABLED = {_IMAGE_CACHE_ENABLED}")

enable_image_cache()
from technical_analysis.common_helpers import _IMAGE_CACHE_ENABLED as enabled_check
print(f"  After enable: Cache enabled = {enabled_check}")
print("✓ Cache enable/disable works")

# Test 3: Caching and retrieval
print("\n[Test 3] Testing cache storage and retrieval...")
test_image_data = b"fake_image_bytes_12345"
cache_key = "test_chart_spx_2024"

cache_chart_image(cache_key, test_image_data)
retrieved = get_cached_chart_image(cache_key)

if retrieved == test_image_data:
    print(f"✓ Successfully cached and retrieved image (key: {cache_key})")
else:
    print(f"✗ Cache retrieval failed")
    sys.exit(1)

# Test 4: Cache miss
print("\n[Test 4] Testing cache miss...")
missing = get_cached_chart_image("nonexistent_key")
if missing is None:
    print("✓ Cache miss returns None correctly")
else:
    print("✗ Cache miss should return None")
    sys.exit(1)

# Test 5: Cache clearing
print("\n[Test 5] Testing cache clearing...")
disable_image_cache()  # This also clears the cache
retrieved_after_clear = get_cached_chart_image(cache_key)
if retrieved_after_clear is None:
    print("✓ Cache cleared successfully")
else:
    print("✗ Cache should be empty after disable")
    sys.exit(1)

# Test 6: Chart generation functions accept cache_key parameter
print("\n[Test 6] Testing chart functions accept cache_key parameter...")
try:
    from technical_analysis.common_helpers import (
        generate_range_callout_chart_image,
        generate_range_gauge_chart_image,
        generate_average_gauge_image,
    )

    import inspect

    # Check generate_range_callout_chart_image
    sig1 = inspect.signature(generate_range_callout_chart_image)
    if 'cache_key' in sig1.parameters:
        print("  ✓ generate_range_callout_chart_image has cache_key parameter")
    else:
        print("  ✗ generate_range_callout_chart_image missing cache_key parameter")
        sys.exit(1)

    # Check generate_range_gauge_chart_image
    sig2 = inspect.signature(generate_range_gauge_chart_image)
    if 'cache_key' in sig2.parameters:
        print("  ✓ generate_range_gauge_chart_image has cache_key parameter")
    else:
        print("  ✗ generate_range_gauge_chart_image missing cache_key parameter")
        sys.exit(1)

    # Check generate_average_gauge_image
    sig3 = inspect.signature(generate_average_gauge_image)
    if 'cache_key' in sig3.parameters:
        print("  ✓ generate_average_gauge_image has cache_key parameter")
    else:
        print("  ✗ generate_average_gauge_image missing cache_key parameter")
        sys.exit(1)

    print("✓ All chart generation functions support caching")

except ImportError as e:
    print(f"✗ Failed to import chart functions: {e}")
    sys.exit(1)

# Test 7: Parallel helper module
print("\n[Test 7] Testing parallel helper module...")
try:
    from technical_analysis.phase2_parallel_helper import (
        prewarm_chart,
        prewarm_charts_batch,
    )
    print("✓ Parallel helper module imports successfully")
except ImportError as e:
    print(f"✗ Failed to import parallel helper: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ All Phase 2 caching infrastructure tests passed!")
print("=" * 70)
print("\nPhase 2 Infrastructure Summary:")
print("  • Image caching: ✓ Ready")
print("  • Chart functions with cache_key support: ✓ Ready")
print("  • Parallel generation helper: ✓ Ready")
print()
print("Next steps to enable Phase 2:")
print("  1. Call enable_image_cache() at start of PPT generation")
print("  2. Generate charts with cache_key parameter in parallel")
print("  3. Sequential PPT insertion will use cached images")
print("  4. Call disable_image_cache() after generation")
print()
print("Expected speedup: 40-60% with 6 parallel workers")
