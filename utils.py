import hashlib
import secrets
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import logging

from config import ALLOWED_IMAGE_EXTENSIONS, MAX_FILE_SIZE_MB

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

    return True, ""

def sanitize_filename(filename: str) -> str:
    path = Path(filename)
    safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', path.stem)
    return f"{safe_name}{path.suffix.lower()}"

def send_verification_email(email: str, token: str):
    verification_link = f"https://app.example.com/verify?token={token}"
    logger.info(f"[EMAIL SIMULATION] Sending verification email to {email}")
    logger.info(f"[EMAIL SIMULATION] Verification link: {verification_link}")
    print(f"\n{'='*60}")
    print(f"VERIFICATION EMAIL SIMULATION")
    print(f"{'='*60}")
    print(f"To: {email}")
    print(f"Subject: Vérifiez votre adresse e-mail")
    print(f"\nCliquez sur ce lien pour vérifier votre compte:")
    print(f"{verification_link}")
    print(f"{'='*60}\n")

def send_password_reset_email(email: str, token: str):
    reset_link = f"https://app.example.com/reset-password?token={token}"
    logger.info(f"[EMAIL SIMULATION] Sending password reset email to {email}")
    logger.info(f"[EMAIL SIMULATION] Reset link: {reset_link}")
    print(f"\n{'='*60}")
    print(f"PASSWORD RESET EMAIL SIMULATION")
    print(f"{'='*60}")
    print(f"To: {email}")
    print(f"Subject: Réinitialisation de votre mot de passe")
    print(f"\nCliquez sur ce lien pour réinitialiser votre mot de passe:")
    print(f"{reset_link}")
    print(f"Ce lien expire dans 1 heure.")
    print(f"{'='*60}\n")

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
