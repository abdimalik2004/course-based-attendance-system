import smtplib
from email.mime.text import MIMEText

EMAIL = "mahadalleabdimalik@gmail.com"
PASSWORD = "ritcnnegzocdpqlq"   # Your 16-character App Password

msg = MIMEText("This is a test email from FastAPI.")
msg["Subject"] = "SMTP Test"
msg["From"] = EMAIL
msg["To"] = "mahadalleabdimalik2@gmail.com"  # Replace with your email

try:
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.ehlo()
    server.starttls()
    server.ehlo()

    server.login(EMAIL, PASSWORD)

    server.sendmail(
        EMAIL,
        msg["To"],
        msg.as_string()
    )

    server.quit()

    print("✅ Email sent successfully!")

except Exception as e:
    print("❌ Error:", e)