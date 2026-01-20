"""Cross-platform HTML-to-image rendering utility.

Works on both Mac and Windows by auto-detecting Chrome/Chromium location.
"""
import os
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from html2image import Html2Image


def get_chrome_path() -> Optional[str]:
    """
    Find Chrome/Chromium executable path for the current platform.

    Returns
    -------
    Optional[str]
        Path to Chrome executable, or None to let html2image auto-detect
    """
    system = platform.system()

    if system == 'Darwin':  # macOS
        paths = [
            '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
            '/Applications/Chromium.app/Contents/MacOS/Chromium',
            '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser',
            os.path.expanduser('~/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'),
        ]
    elif system == 'Windows':
        paths = [
            'C:/Program Files/Google/Chrome/Application/chrome.exe',
            'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
            os.path.expandvars('%LOCALAPPDATA%/Google/Chrome/Application/chrome.exe'),
            'C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe',
        ]
    else:  # Linux
        paths = [
            '/usr/bin/google-chrome',
            '/usr/bin/chromium',
            '/usr/bin/chromium-browser',
            '/snap/bin/chromium',
        ]

    for path in paths:
        if os.path.exists(path):
            return path

    return None  # Let html2image try to auto-detect


def render_html_to_image(
    html_content: str,
    output_path: str,
    size: Tuple[int, int],
    filename: str = "output.png"
) -> str:
    """
    Render HTML content to a PNG image (cross-platform).

    Parameters
    ----------
    html_content : str
        The HTML string to render
    output_path : str
        Full path where the output PNG should be saved
    size : Tuple[int, int]
        Width and height of the output image in pixels (width, height)
    filename : str
        Temporary filename used during rendering

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

    print(f"[HTML2Image] Starting render...")
    print(f"[HTML2Image] Output: {output_path}")
    print(f"[HTML2Image] Size: {width}x{height}px")
    print(f"[HTML2Image] Platform: {platform.system()}")

    # Find Chrome executable
    chrome_path = get_chrome_path()
    if chrome_path:
        print(f"[HTML2Image] Chrome found: {chrome_path}")
    else:
        print(f"[HTML2Image] Chrome not found, using auto-detect")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"[HTML2Image] Temp dir: {tmpdir}")

            # Create Html2Image instance with Chrome path if found
            if chrome_path:
                hti = Html2Image(
                    output_path=tmpdir,
                    size=(width, height),
                    browser_executable=chrome_path
                )
            else:
                hti = Html2Image(
                    output_path=tmpdir,
                    size=(width, height)
                )

            # Render the HTML
            print(f"[HTML2Image] Rendering HTML ({len(html_content)} chars)...")
            hti.screenshot(html_str=html_content, save_as=filename)

            # Check if file was created
            temp_output = Path(tmpdir) / filename
            if not temp_output.exists():
                raise RuntimeError(f"Image file was not created at {temp_output}")

            file_size = temp_output.stat().st_size
            print(f"[HTML2Image] Image created: {file_size} bytes")

            if file_size == 0:
                raise RuntimeError("Image file is empty (0 bytes)")

            # Move to final destination
            shutil.move(str(temp_output), output_path)
            print(f"[HTML2Image] Moved to: {output_path}")

            # Verify final file
            if not Path(output_path).exists():
                raise RuntimeError(f"Failed to move image to {output_path}")

            print(f"[HTML2Image] Render successful!")
            return output_path

    except Exception as e:
        print(f"[HTML2Image] ERROR: Render failed!")
        print(f"[HTML2Image] Error type: {type(e).__name__}")
        print(f"[HTML2Image] Error message: {e}")

        # Provide helpful troubleshooting info
        print(f"\n[HTML2Image] TROUBLESHOOTING:")
        print(f"  1. Ensure Chrome/Chromium is installed")
        if platform.system() == 'Darwin':
            print(f"     Mac: Install from https://www.google.com/chrome/")
        elif platform.system() == 'Windows':
            print(f"     Windows: Install from https://www.google.com/chrome/")
        print(f"  2. Try running: pip install --upgrade html2image")
        print(f"  3. Check temp directory permissions: {tempfile.gettempdir()}")

        import traceback
        traceback.print_exc()
        raise
