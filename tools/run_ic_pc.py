"""
IC2 Market Compass — intern PC entry point.

Invoked by Windows Task Scheduler every Wednesday 07:15 (after HTEI's 06:50
Bloomberg login). Performs:

    1. Resolve dates: price = last business day, committee = today (Wed).
       A one-liner date in ${IC_DROPBOX_PATH}/.committee_override overrides
       the committee date (used for Thursday shifts). The file is consumed
       on use and deleted on pipeline success.
    2. Verify Bloomberg is reachable on BLOOMBERG_HOST:BLOOMBERG_PORT.
       If not, call HTEI's bloomberg_login.py to re-login, wait, retry.
       On retry failure, email the failure and exit.
    3. Run run_ic.py --stage full with resolved dates.
    4. On success, email jcourtial with the Dropbox path to the PPTX.

Config via environment (see .env on the intern PC):

    IC_DROPBOX_PATH         e.g. C:\\Users\\User3\\Dropbox\\...\\ic
    BLOOMBERG_HOST          localhost
    BLOOMBERG_PORT          8194
    HTEI_BLOOMBERG_LOGIN    path to HTEI's bloomberg_login.py
    SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASS / RECIPIENT
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

# Allow `from tools.ic_email import ...` and `from pipeline.bloomberg import ...`
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.ic_email import notify_failure, notify_success  # noqa: E402


def _prev_business_day(d: date) -> date:
    """Return the most recent weekday strictly before d. Does not skip holidays."""
    n = d - timedelta(days=1)
    while n.weekday() >= 5:
        n -= timedelta(days=1)
    return n


def _resolve_dates() -> tuple[str, str]:
    """Return (price_date, committee_date) as YYYY-MM-DD strings.

    committee_date: today, unless a .committee_override file exists in the
    IC Dropbox folder containing a YYYY-MM-DD date. That file is deleted
    after a successful run (by the caller).
    """
    today = date.today()
    committee = today

    override_path = Path(
        os.environ.get("IC_DROPBOX_PATH", ".")
    ) / ".committee_override"
    if override_path.exists():
        try:
            committee = datetime.strptime(
                override_path.read_text().strip(), "%Y-%m-%d"
            ).date()
            log.info("Committee date overridden to %s via %s", committee, override_path)
        except ValueError as e:
            log.warning("Ignoring malformed %s: %s", override_path, e)

    price = _prev_business_day(committee)
    return price.isoformat(), committee.isoformat()


def _bloomberg_ok() -> bool:
    """Lightweight check: can we start a blpapi session to the configured host?"""
    try:
        import blpapi
    except ImportError:
        log.error("blpapi not installed in this Python environment")
        return False

    opts = blpapi.SessionOptions()
    opts.setServerHost(os.environ.get("BLOOMBERG_HOST", "localhost"))
    opts.setServerPort(int(os.environ.get("BLOOMBERG_PORT", "8194")))
    session = blpapi.Session(opts)
    try:
        if not session.start():
            return False
        ok = session.openService("//blp/refdata")
        return bool(ok)
    finally:
        try:
            session.stop()
        except Exception:
            pass


def _try_relogin_bloomberg() -> bool:
    """Call HTEI's bloomberg_login.py and re-check. Returns True if Bloomberg is up."""
    login_script = os.environ.get("HTEI_BLOOMBERG_LOGIN")
    if not login_script or not Path(login_script).exists():
        log.error(
            "HTEI_BLOOMBERG_LOGIN is not set or file missing (%s)", login_script
        )
        return False

    log.warning("Bloomberg down — invoking %s", login_script)
    try:
        subprocess.run(
            [sys.executable, login_script],
            check=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        log.error("bloomberg_login.py timed out")
        return False
    except subprocess.CalledProcessError as e:
        log.error("bloomberg_login.py exited with %s", e.returncode)
        return False

    # Give DAPI a few seconds to come up
    for attempt in range(6):
        time.sleep(5)
        if _bloomberg_ok():
            log.info("Bloomberg reachable after re-login (attempt %d)", attempt + 1)
            return True
    return False


def _clear_override() -> None:
    p = Path(os.environ.get("IC_DROPBOX_PATH", ".")) / ".committee_override"
    if p.exists():
        try:
            p.unlink()
            log.info("Removed %s", p)
        except OSError as e:
            log.warning("Could not remove %s: %s", p, e)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    price_date, committee_date = _resolve_dates()
    log.info("price_date=%s  committee_date=%s", price_date, committee_date)

    # 1. Bloomberg connectivity (with single retry via HTEI login)
    if not _bloomberg_ok():
        log.warning("Bloomberg not reachable on first check")
        notify_failure(
            "bloomberg-precheck",
            "Bloomberg was not reachable at pipeline start; attempting re-login.",
        )
        if not _try_relogin_bloomberg():
            notify_failure(
                "bloomberg-relogin",
                "bloomberg_login.py retry failed; pipeline aborted.",
            )
            return 2

    # 2. Run the pipeline
    dropbox = Path(os.environ.get("IC_DROPBOX_PATH", "."))
    pptx_path = dropbox / f"Market_Compass_{committee_date.replace('-', '')}.pptx"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "run_ic.py"),
        "--stage", "full",
        "--date", price_date,
        "--committee-date", committee_date,
    ]
    log.info("Running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    except subprocess.CalledProcessError as e:
        notify_failure(
            "run_ic",
            f"run_ic.py exited with {e.returncode}. See pipeline logs on the intern PC.",
        )
        return 3

    # 3. Verify output exists and notify
    if not pptx_path.exists():
        notify_failure(
            "output-missing",
            f"Pipeline reported success but {pptx_path} is missing.",
        )
        return 4

    notify_success(pptx_path, committee_date, price_date)
    _clear_override()
    return 0


if __name__ == "__main__":
    sys.exit(main())
