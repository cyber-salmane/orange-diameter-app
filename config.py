import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "admin.db"
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# SECURITY: Set ADMIN_PASSWORD environment variable in production!
# Default password is insecure and should NEVER be used in production.
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "ORANGEADMIN")

MAX_FILE_SIZE_MB = 10
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

SESSION_TIMEOUT_HOURS = 24
MAX_LOGIN_ATTEMPTS = 5
LOGIN_ATTEMPT_WINDOW_MINUTES = 15

CLEANUP_OLD_FILES_DAYS = 30

# NOTE: Email verification is simulated (prints to console).
# For production, implement real email sending in utils.py
EMAIL_VERIFICATION_ENABLED = True
