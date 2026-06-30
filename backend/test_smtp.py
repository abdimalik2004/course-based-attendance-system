"""
Quick SMTP connection test — run this from the backend folder to verify
your Gmail App Password is working before restarting the server.

Usage:
  cd backend
  python test_smtp.py
"""
import os
import smtplib
from pathlib import Path
from dotenv import load_dotenv

# Load the same env files the app uses
_dir = Path(__file__).parent
load_dotenv(_dir / ".env", override=False)
env = os.getenv("APP_ENV", "development")
load_dotenv(_dir / f".env.{env}", override=True)

host     = os.getenv("SMTP_HOST", "smtp.gmail.com")
port     = int(os.getenv("SMTP_PORT", "587"))
email    = os.getenv("SMTP_EMAIL", "")
password = os.getenv("SMTP_PASSWORD", "")

if not email or not password:
    print("❌  SMTP_EMAIL or SMTP_PASSWORD is not set in .env.development")
    exit(1)

print(f"Testing SMTP connection to {host}:{port} as {email} ...")

try:
    with smtplib.SMTP(host, port, timeout=10) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login(email, password)
        print("✅  SMTP login successful! Emails will be sent correctly.")
except smtplib.SMTPAuthenticationError:
    print("❌  Authentication failed.")
    print("   → Make sure SMTP_PASSWORD is a Gmail App Password (16 chars),")
    print("     NOT your regular Gmail password.")
    print("   → Generate one at: myaccount.google.com/apppasswords")
except Exception as e:
    print(f"❌  Connection error: {e}")
