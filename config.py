import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

DEFAULT_DB_PATH = Path(os.environ.get("DB_PATH", ""))
if not DEFAULT_DB_PATH or str(DEFAULT_DB_PATH) == ".":
    DEFAULT_DB_PATH = BASE_DIR / "admin.db"
DB_PATH = DEFAULT_DB_PATH
UPLOADS_DIR = Path(os.environ.get("UPLOADS_DIR", BASE_DIR / "uploads"))
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")

MAX_FILE_SIZE_MB = 10
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ALLOWED_IMAGE_MIME_TYPES = {"image/png", "image/jpeg"}

SESSION_TIMEOUT_HOURS = 24
MAX_LOGIN_ATTEMPTS = 5
LOGIN_ATTEMPT_WINDOW_MINUTES = 15
MAX_REGISTRATION_ATTEMPTS = 5
REGISTRATION_ATTEMPT_WINDOW_MINUTES = 15
MAX_PASSWORD_RESET_REQUESTS = 3
PASSWORD_RESET_WINDOW_MINUTES = 60

CLEANUP_OLD_FILES_DAYS = 30

EMAIL_VERIFICATION_ENABLED = True

# App base URL used in verification emails
APP_BASE_URL = os.environ.get("APP_BASE_URL", "http://localhost:7860")

# SMTP Configuration for real email sending
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_FROM_EMAIL = os.environ.get("SMTP_FROM_EMAIL", "noreply@orange-diameter.com")
USE_REAL_SMTP = bool(SMTP_USERNAME and SMTP_PASSWORD)

# SendGrid API configuration for reliable transactional email
SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY", "")
SENDGRID_FROM_EMAIL = os.environ.get("SENDGRID_FROM_EMAIL", SMTP_FROM_EMAIL)
USE_SENDGRID = bool(SENDGRID_API_KEY)
