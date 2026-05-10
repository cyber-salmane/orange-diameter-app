import hashlib
import secrets
import re
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from io import BytesIO

import requests
from PIL import Image

from config import (
    ALLOWED_IMAGE_EXTENSIONS, MAX_FILE_SIZE_MB, ALLOWED_IMAGE_MIME_TYPES,
    SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SMTP_FROM_EMAIL,
    USE_REAL_SMTP, APP_BASE_URL, SENDGRID_API_KEY, SENDGRID_FROM_EMAIL, USE_SENDGRID
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_token(length: int = 32) -> str:
    return secrets.token_urlsafe(length)

def validate_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_username(username: str) -> bool:
    if len(username) < 3 or len(username) > 30:
        return False
    return bool(re.match(r'^[a-zA-Z0-9_-]+$', username))

def validate_password(password: str) -> tuple[bool, str]:
    if len(password) < 8:
        return False, "Le mot de passe doit contenir au moins 8 caractères."
    if not re.search(r'[A-Z]', password):
        return False, "Le mot de passe doit contenir au moins une majuscule."
    if not re.search(r'[a-z]', password):
        return False, "Le mot de passe doit contenir au moins une minuscule."
    if not re.search(r'[0-9]', password):
        return False, "Le mot de passe doit contenir au moins un chiffre."
    return True, ""

def validate_file(filename: str, file_size_bytes: int) -> tuple[bool, str]:
    path = Path(filename)
    ext = path.suffix.lower()

    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return False, f"Type de fichier non autorisé. Formats acceptés: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"

    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size_bytes > max_bytes:
        return False, f"Fichier trop volumineux. Taille maximale: {MAX_FILE_SIZE_MB}MB"

    try:
        with Image.open(path) as img:
            if img.format and img.get_format_mimetype() not in ALLOWED_IMAGE_MIME_TYPES:
                return False, f"Type de fichier non autorisé. Formats acceptés: {', '.join(ALLOWED_IMAGE_MIME_TYPES)}"
            img.verify()
    except Exception:
        return False, "Le fichier n'est pas une image valide."

    return True, ""


def is_valid_image(image) -> bool:
    try:
        if image is None:
            return False

        if isinstance(image, Image.Image):
            buffer = BytesIO()
            image.save(buffer, format=image.format or "PNG")
            buffer.seek(0)
            with Image.open(buffer) as img:
                return img.format in {"PNG", "JPEG"} and img.get_format_mimetype() in ALLOWED_IMAGE_MIME_TYPES

        if isinstance(image, (bytes, bytearray)):
            with Image.open(BytesIO(image)) as img:
                return img.format in {"PNG", "JPEG"} and img.get_format_mimetype() in ALLOWED_IMAGE_MIME_TYPES

        return False
    except Exception:
        return False


def sanitize_filename(filename: str) -> str:
    path = Path(filename)
    safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', path.stem)
    return f"{safe_name}{path.suffix.lower()}"


def _build_email_html(title: str, button_text: str, button_url: str, fallback_url: str, message: str) -> str:
    return f"""<html>
  <body style='font-family:Arial,Helvetica,sans-serif;background:#f3f4f6;color:#0f172a;margin:0;padding:20px'>
    <table width='100%' cellpadding='0' cellspacing='0' style='max-width:600px;margin:0 auto;background:#ffffff;border-radius:16px;overflow:hidden;box-shadow:0 20px 50px rgba(15,23,42,0.08)'>
      <tr style='background:#2563eb;color:#ffffff'>
        <td style='padding:30px;text-align:center'>
          <h1 style='margin:0;font-size:24px'>{title}</h1>
        </td>
      </tr>
      <tr>
        <td style='padding:32px'>
          <p style='font-size:16px;line-height:1.7;color:#334155'>{message}</p>
          <div style='text-align:center;margin:32px 0'>
            <a href='{button_url}' style='display:inline-block;padding:14px 26px;background:#2563eb;color:#ffffff;border-radius:12px;text-decoration:none;font-weight:700'>{button_text}</a>
          </div>
          <p style='font-size:14px;line-height:1.7;color:#475569'>Si le bouton ne fonctionne pas, copiez-collez ce lien dans votre navigateur :</p>
          <p style='font-size:14px;line-height:1.7;color:#2563eb;word-break:break-all'><a href='{fallback_url}'>{fallback_url}</a></p>
          <p style='font-size:13px;line-height:1.7;color:#64748b;margin-top:28px'>Si vous n'avez pas demandé cette action, vous pouvez ignorer cet e-mail.</p>
        </td>
      </tr>
    </table>
  </body>
</html>"""


def _sendgrid_email(to_email: str, subject: str, html: str, text: str) -> tuple[bool, str]:
    headers = {
        'Authorization': f'Bearer {SENDGRID_API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        'personalizations': [{
            'to': [{'email': to_email}]
        }],
        'from': {'email': SENDGRID_FROM_EMAIL},
        'subject': subject,
        'content': [
            {'type': 'text/plain', 'value': text},
            {'type': 'text/html', 'value': html}
        ]
    }
    try:
        response = requests.post('https://api.sendgrid.com/v3/mail/send', headers=headers, json=payload, timeout=15)
        if response.status_code in (200, 202):
            logger.info(f"Email sent successfully to {to_email}")
            return True, ""
        error_message = response.text.strip() or response.reason
        logger.error(f"SendGrid error: {response.status_code} {error_message}")
        return False, f"SendGrid error: {error_message}"
    except Exception as e:
        logger.error(f"SendGrid error: {e}")
        return False, f"SendGrid error: {e}"


def _smtp_email(to_email: str, subject: str, html: str, text: str) -> tuple[bool, str]:
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = SMTP_FROM_EMAIL
        msg['To'] = to_email

        part1 = MIMEText(text, 'plain')
        part2 = MIMEText(html, 'html')
        msg.attach(part1)
        msg.attach(part2)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)

        logger.info(f"Email sent successfully to {to_email}")
        return True, ""
    except Exception as e:
        logger.error(f"SMTP error: {e}")
        return False, f"SMTP error: {e}"


def send_email(to_email: str, subject: str, text: str, html: str) -> tuple[bool, str]:
    if USE_SENDGRID:
        return _sendgrid_email(to_email, subject, html, text)

    if USE_REAL_SMTP:
        return _smtp_email(to_email, subject, html, text)

    logger.info(f"[EMAIL SIMULATION] Sending email to {to_email}")
    logger.info(f"[EMAIL SIMULATION] Subject: {subject}")
    logger.info(f"[EMAIL SIMULATION] HTML content: {html}")
    return True, ""


def send_verification_email(email: str, token: str) -> tuple[bool, str]:
    verification_link = f"/verify?token={token}"
    fallback_full_link = f"{APP_BASE_URL}/verify?token={token}"
    text = f"Cliquez sur ce lien pour vérifier votre compte: {fallback_full_link}\nSi vous n'avez pas demandé cet e-mail, ignorez-le."
    html = _build_email_html(
        title="Vérifiez votre adresse e-mail",
        button_text="Vérifier mon compte",
        button_url=fallback_full_link,
        fallback_url=fallback_full_link,
        message="Cliquez sur le bouton ci-dessous pour vérifier votre adresse e-mail et activer votre compte."
    )
    return send_email(email, 'Vérification de votre adresse e-mail', text, html)


def send_password_reset_email(email: str, token: str) -> tuple[bool, str]:
    reset_link = f"/reset-password?token={token}"
    fallback_full_link = f"{APP_BASE_URL}/reset-password?token={token}"
    text = f"Cliquez sur ce lien pour réinitialiser votre mot de passe: {fallback_full_link}\nCe lien expire dans 1 heure. Si vous n'avez pas demandé cette action, ignorez cet e-mail."
    html = _build_email_html(
        title="Réinitialisation de votre mot de passe",
        button_text="Réinitialiser mon mot de passe",
        button_url=fallback_full_link,
        fallback_url=fallback_full_link,
        message="Cliquez sur le bouton ci-dessous pour réinitialiser votre mot de passe. Ce lien expire dans 1 heure."
    )
    return send_email(email, 'Réinitialisation de votre mot de passe', text, html)

def is_token_expired(created_at: str, expiry_hours: int = 1) -> bool:
    try:
        created = datetime.fromisoformat(created_at)
        expiry = created + timedelta(hours=expiry_hours)
        return datetime.now() > expiry
    except:
        return True

def cleanup_old_files(uploads_dir: Path, days: int):
    try:
        cutoff = datetime.now() - timedelta(days=days)
        deleted_count = 0

        for file_path in uploads_dir.glob("*"):
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff:
                    file_path.unlink()
                    deleted_count += 1
                    logger.info(f"Deleted old file: {file_path.name}")

        if deleted_count > 0:
            logger.info(f"Cleanup completed: {deleted_count} file(s) deleted")

        return deleted_count
    except Exception as e:
        logger.error(f"Error during file cleanup: {e}")
        return 0
