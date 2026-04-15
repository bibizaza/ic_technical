"""
SMTP email helper for the IC2 pipeline (intern PC runtime).

Reads configuration from environment:
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, RECIPIENT

Mirrors the pattern used by HTEI so both pipelines share the same Gmail
app password and SMTP setup.
"""
from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

log = logging.getLogger(__name__)


def _smtp_config() -> dict:
    return {
        "host":      os.environ.get("SMTP_HOST", "smtp.gmail.com"),
        "port":      int(os.environ.get("SMTP_PORT", "587")),
        "user":      os.environ["SMTP_USER"],
        "password":  os.environ["SMTP_PASS"],
        "recipient": os.environ.get("RECIPIENT", "jcourtial@herculis.ch"),
    }


def send_email(subject: str, body: str, attachment: Path | None = None) -> None:
    """Send a plain-text email. If attachment is given, include it as a file.

    Raises on SMTP errors so the caller can record a failure in status.
    """
    cfg = _smtp_config()

    msg = EmailMessage()
    msg["From"] = cfg["user"]
    msg["To"] = cfg["recipient"]
    msg["Subject"] = subject
    msg.set_content(body)

    if attachment is not None and attachment.exists():
        data = attachment.read_bytes()
        msg.add_attachment(
            data,
            maintype="application",
            subtype="octet-stream",
            filename=attachment.name,
        )

    with smtplib.SMTP(cfg["host"], cfg["port"], timeout=30) as s:
        s.starttls()
        s.login(cfg["user"], cfg["password"])
        s.send_message(msg)

    log.info("Email sent to %s (subject=%r)", cfg["recipient"], subject)


def notify_success(pptx_path: Path, committee_date: str, price_date: str) -> None:
    subject = f"IC Market Compass generated — {committee_date}"
    body = (
        f"The IC Market Compass presentation has been generated.\n"
        f"\n"
        f"Committee date: {committee_date}\n"
        f"Price reference: {price_date}\n"
        f"\n"
        f"File: {pptx_path}\n"
    )
    send_email(subject, body)


def notify_failure(stage: str, error: str) -> None:
    subject = f"IC Market Compass FAILED at {stage}"
    body = (
        f"The IC Market Compass pipeline failed.\n"
        f"\n"
        f"Stage: {stage}\n"
        f"Error: {error}\n"
    )
    try:
        send_email(subject, body)
    except Exception as e:
        log.error("Failure email could not be sent: %s", e)
