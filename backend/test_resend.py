"""
Quick Resend test — run from the backend folder:
  python test_resend.py your@email.com
"""
import os, sys
from pathlib import Path
from dotenv import load_dotenv

_dir = Path(__file__).parent
load_dotenv(_dir / ".env", override=False)
load_dotenv(_dir / f".env.{os.getenv('APP_ENV','development')}", override=True)

api_key = os.getenv("RESEND_API_KEY", "")
if not api_key:
    print("❌  RESEND_API_KEY not set in .env.development")
    sys.exit(1)

to_email = sys.argv[1] if len(sys.argv) > 1 else "mahadalleabdimalik@gmail.com"
print(f"API key loaded: {api_key[:8]}...")
print(f"Sending test email to: {to_email}")

try:
    import resend
    resend.api_key = api_key
    result = resend.Emails.send({
        "from": "Heegan Attendance <onboarding@resend.dev>",
        "to": [to_email],
        "subject": "Resend test",
        "html": "<p>Test email from Heegan. If you see this, Resend is working!</p>",
    })
    print(f"✅  Sent! Response: {result}")
except Exception as e:
    print(f"❌  Error: {e}")
