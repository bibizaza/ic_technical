"""Cross-platform HTML-to-image rendering utility using Playwright.

Uses Playwright's bundled Chromium for consistent rendering across Mac, Windows, and Linux.
"""
import os
import platform
from pathlib import Path
from typing import Tuple

from playwright.sync_api import sync_playwright


def render_html_to_image(
    html_content: str,
    output_path: str,
    size: Tuple[int, int],
    filename: str = "output.png",
    device_scale_factor: int = 1
) -> str:
    """
    Render HTML content to a PNG image using Playwright.

    Uses Playwright's bundled Chromium browser for consistent cross-platform rendering.
    The device_scale_factor parameter enables HiDPI rendering for crisp output.

    Parameters
    ----------
    html_content : str
        The HTML string to render
    output_path : str
        Full path where the output PNG should be saved
    size : Tuple[int, int]
        Viewport width and height in CSS pixels (width, height)
    filename : str
        Not used with Playwright (kept for backward compatibility)
    device_scale_factor : int
        Device scale factor for HiDPI rendering (default 1, use 2-4 for crisp output)

    Returns
    -------
    str
        Path to the saved PNG image

    Raises
    ------
    RuntimeError
        If image rendering fails
    """
    width, height = size

    print(f"[Playwright] Starting render...")
    print(f"[Playwright] Output: {output_path}")
    print(f"[Playwright] Viewport: {width}x{height}px")
    print(f"[Playwright] Scale factor: {device_scale_factor}x")
    print(f"[Playwright] Platform: {platform.system()}")

    try:
        with sync_playwright() as p:
            # Launch headless Chromium (bundled with Playwright)
            print(f"[Playwright] Launching Chromium...")
            browser = p.chromium.launch(headless=True)

            # Create page with HiDPI scaling
            page = browser.new_page(
                viewport={'width': width, 'height': height},
                device_scale_factor=device_scale_factor
            )

            # Load HTML content
            print(f"[Playwright] Loading HTML ({len(html_content)} chars)...")
            page.set_content(html_content, wait_until='networkidle')

            # Brief wait for any final rendering
            page.wait_for_timeout(100)

            # Take screenshot
            print(f"[Playwright] Taking screenshot...")
            page.screenshot(path=output_path, full_page=True)

            # Cleanup
            browser.close()

        # Verify output
        if not Path(output_path).exists():
            raise RuntimeError(f"Image file was not created at {output_path}")

        file_size = Path(output_path).stat().st_size
        if file_size == 0:
            raise RuntimeError("Image file is empty (0 bytes)")

        print(f"[Playwright] Image saved: {output_path} ({file_size:,} bytes)")
        return output_path

    except Exception as e:
        print(f"[Playwright] ERROR: Render failed!")
        print(f"[Playwright] Error type: {type(e).__name__}")
        print(f"[Playwright] Error message: {e}")

        # Provide helpful troubleshooting info
        print(f"\n[Playwright] TROUBLESHOOTING:")
        print(f"  1. Ensure Playwright is installed: pip install playwright")
        print(f"  2. Install Chromium browser: playwright install chromium")
        print(f"  3. Or install all browsers: playwright install")

        import traceback
        traceback.print_exc()
        raise
