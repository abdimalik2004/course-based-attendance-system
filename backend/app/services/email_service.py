"""
email_service.py
────────────────
Provides two things:

1. OTPStore  — thread-safe in-memory store for 6-digit password-reset codes.
               No database migration required; codes expire automatically and
               are cleared when the server restarts (which is fine for
               short-lived password-reset flows).

2. send_reset_code()  — sends an HTML email with the reset code.
   Sending priority:
     1. Resend API  (set RESEND_API_KEY in backend/.env)
     2. Gmail SMTP  (set SMTP_EMAIL + SMTP_PASSWORD in backend/.env)
   At least one must be configured in production.
"""

from __future__ import annotations

import logging
import random
import secrets
import smtplib
import string
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from threading import Lock

from app.core.config import settings

logger = logging.getLogger(__name__)


# ── In-memory OTP store ──────────────────────────────────────────────────────

class OTPStore:
    """
    Thread-safe store for password-reset OTP codes.

    Lifecycle of one entry:
      create()     → code + 10-min expiry stored under `email`
      verify()     → code checked; on success a reset_token is issued (15-min expiry)
      consume()    → reset_token exchanged for the email; entry deleted
    """

    def __init__(self) -> None:
        self._store: dict[str, dict] = {}
        self._lock = Lock()

    # ── Public API ───────────────────────────────────────────────────────────

    def create(self, email: str, code: str, expires_minutes: int = 10) -> None:
        """Store a new OTP code for `email`, overwriting any existing entry."""
        with self._lock:
            self._store[email.lower()] = {
                "code": code,
                "expires_at": datetime.utcnow() + timedelta(minutes=expires_minutes),
                "attempts": 0,
                "reset_token": None,
                "reset_expires_at": None,
            }

    def verify(self, email: str, code: str) -> str | None:
        """
        Validate `code` for `email`.

        Returns a one-time `reset_token` (URL-safe string) on success,
        or None if the code is wrong / expired / too many attempts.
        The code cannot be reused after a successful verification.
        """
        key = email.lower()
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None

            # Expired
            if datetime.utcnow() > entry["expires_at"]:
                del self._store[key]
                return None

            # Already verified (code already consumed, waiting for password set)
            if entry.get("reset_token") is not None:
                # Still within reset window — re-issue the same token
                if datetime.utcnow() <= entry["reset_expires_at"]:
                    return entry["reset_token"]
                # Reset window also expired
                del self._store[key]
                return None

            # Too many wrong attempts
            if entry["attempts"] >= 5:
                del self._store[key]
                return None

            # Wrong code
            if entry["code"] != code.strip():
                entry["attempts"] += 1
                return None

            # ✅ Correct — generate a reset token valid for 15 minutes
            reset_token = secrets.token_urlsafe(32)
            entry["reset_token"] = reset_token
            entry["reset_expires_at"] = datetime.utcnow() + timedelta(minutes=15)
            entry.pop("code", None)   # code can't be reused
            return reset_token

    def consume_reset_token(self, reset_token: str) -> str | None:
        """
        Exchange `reset_token` for the associated email, then delete the entry.
        Returns the email on success, None if the token is unknown or expired.
        """
        with self._lock:
            for email, entry in list(self._store.items()):
                if entry.get("reset_token") == reset_token:
                    if entry.get("reset_expires_at") and datetime.utcnow() > entry["reset_expires_at"]:
                        del self._store[email]
                        return None
                    del self._store[email]
                    return email
        return None

    def remaining_attempts(self, email: str) -> int:
        """How many code-entry attempts remain for this email (5 max)."""
        with self._lock:
            entry = self._store.get(email.lower())
            if not entry:
                return 0
            return max(0, 5 - entry.get("attempts", 0))


# Module-level singleton
otp_store = OTPStore()


# ── Code generation ───────────────────────────────────────────────────────────

def generate_otp(length: int = 6) -> str:
    """Return a zero-padded numeric OTP of `length` digits."""
    return "".join(random.choices(string.digits, k=length))


# ── Email sending ─────────────────────────────────────────────────────────────

def _build_html(code: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#0b0f19;font-family:'Segoe UI',Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0b0f19;padding:40px 0;">
    <tr>
      <td align="center">
        <table width="480" cellpadding="0" cellspacing="0"
               style="background:#131928;border-radius:16px;border:1px solid rgba(255,255,255,0.08);overflow:hidden;">
          <tr>
            <td style="height:4px;background:linear-gradient(90deg,#2563eb,#3b82f6);"></td>
          </tr>
          <tr>
            <td style="padding:40px 48px 32px;">
              <p style="margin:0 0 8px;font-size:22px;font-weight:700;color:#ffffff;">
                Reset your password
              </p>
              <p style="margin:0 0 32px;font-size:14px;color:#9ca3af;">
                Use the code below to reset your Heegan account password.
                It expires in <strong style="color:#f3f4f6;">10 minutes</strong>.
              </p>
              <div style="text-align:center;margin:0 0 32px;">
                <div style="display:inline-block;background:#1e2d4a;border:1px solid #2563eb;
                            border-radius:12px;padding:20px 40px;">
                  <span style="font-size:40px;font-weight:800;letter-spacing:12px;
                               color:#3b82f6;font-family:monospace;">{code}</span>
                </div>
              </div>
              <p style="margin:0;font-size:13px;color:#6b7280;text-align:center;">
                If you didn't request a password reset, you can safely ignore this email.
              </p>
            </td>
          </tr>
          <tr>
            <td style="padding:16px 48px;border-top:1px solid rgba(255,255,255,0.06);">
              <p style="margin:0;font-size:12px;color:#4b5563;text-align:center;">
                Heegan Attendance System &nbsp;·&nbsp; Zamzam University
              </p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""


def _send_via_resend(to_email: str, subject: str, html: str) -> bool:
    """Send using the Resend API. Returns True on success."""
    try:
        import resend  # type: ignore
    except ImportError:
        logger.error("resend package not installed — run: pip install resend")
        return False

    resend.api_key = settings.resend_api_key
    from_addr = settings.resend_from or f"{settings.smtp_from_name} <onboarding@resend.dev>"

    try:
        resend.Emails.send({
            "from": from_addr,
            "to": [to_email],
            "subject": subject,
            "html": html,
        })
        logger.info("Password-reset code sent to %s via Resend", to_email)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Resend error sending to %s: %s", to_email, exc)
        return False


def _send_via_smtp(to_email: str, subject: str, html: str) -> bool:
    """Send using Gmail SMTP. Returns True on success."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{settings.smtp_from_name} <{settings.smtp_email}>"
    msg["To"] = to_email
    msg.attach(MIMEText(html, "html", "utf-8"))

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(settings.smtp_email, settings.smtp_password)
            smtp.sendmail(settings.smtp_email, to_email, msg.as_string())
        logger.info("Password-reset code sent to %s via SMTP", to_email)
        return True
    except smtplib.SMTPAuthenticationError:
        logger.error(
            "SMTP authentication failed for %s — check SMTP_EMAIL and SMTP_PASSWORD "
            "(use a Gmail App Password, not your regular password)",
            settings.smtp_email,
        )
    except smtplib.SMTPException as exc:
        logger.error("SMTP error sending to %s: %s", to_email, exc)
    except OSError as exc:
        logger.error("Network error sending reset email to %s: %s", to_email, exc)
    return False


def send_reset_code(to_email: str, code: str) -> bool:
    """
    Send a password-reset code email to `to_email`.

    Tries Resend first (if RESEND_API_KEY is set), then falls back to SMTP.
    Returns True on success, False if neither is configured or sending fails.
    """
    subject = "Your Heegan password reset code"
    html = _build_html(code)

    # 1 — Resend API
    if settings.resend_api_key:
        return _send_via_resend(to_email, subject, html)

    # 2 — SMTP fallback
    if settings.smtp_email and settings.smtp_password:
        return _send_via_smtp(to_email, subject, html)

    logger.warning(
        "No email provider configured — set RESEND_API_KEY or SMTP_EMAIL+SMTP_PASSWORD "
        "in backend/.env. Reset code for %s: %s",
        to_email, code,
    )
    return False
