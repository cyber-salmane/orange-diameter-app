import uuid
import hashlib
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
import logging

from db import get_db, get_user_by_username, get_user_by_email
from utils import (
    validate_email, validate_username, validate_password,
    generate_token, send_verification_email, send_password_reset_email,
    is_token_expired
)
from config import (
    SESSION_TIMEOUT_HOURS, MAX_LOGIN_ATTEMPTS,
    LOGIN_ATTEMPT_WINDOW_MINUTES, EMAIL_VERIFICATION_ENABLED,
    MAX_REGISTRATION_ATTEMPTS, REGISTRATION_ATTEMPT_WINDOW_MINUTES,
    MAX_PASSWORD_RESET_REQUESTS, PASSWORD_RESET_WINDOW_MINUTES
)

logger = logging.getLogger(__name__)

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    except:
        return False


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


def _log_action(table: str, email: str, ip: str, success: bool):
    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    conn = get_db()
    try:
        conn.execute(
            f"INSERT INTO {table} (id, email, ip, success, timestamp) VALUES (?,?,?,?,?)",
            (str(uuid.uuid4()), email, ip, 1 if success else 0, now)
        )
        conn.commit()
    finally:
        conn.close()


def _check_rate_limit(table: str, email: str, ip: str, max_attempts: int, window_minutes: int) -> Tuple[bool, str]:
    now = datetime.now()
    cutoff = (now - timedelta(minutes=window_minutes)).isoformat(sep=" ", timespec="seconds")
    conn = get_db()
    try:
        attempts = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE (email=? OR ip=?) AND timestamp > ? AND success=0",
            (email.lower(), ip, cutoff)
        ).fetchone()[0]
    finally:
        conn.close()

    if attempts >= max_attempts:
        return False, f"Trop de tentatives. Réessayez dans {window_minutes} minutes."
    return True, ""


def check_rate_limit(username: str, ip: str) -> Tuple[bool, str]:
    now = datetime.now()
    cutoff = (now - timedelta(minutes=LOGIN_ATTEMPT_WINDOW_MINUTES)).isoformat(sep=" ", timespec="seconds")

    conn = get_db()
    attempts = conn.execute(
        "SELECT COUNT(*) FROM login_attempts WHERE (username=? OR ip=?) AND timestamp > ? AND success=0",
        (username, ip, cutoff)
    ).fetchone()[0]
    conn.close()

    if attempts >= MAX_LOGIN_ATTEMPTS:
        return False, f"Trop de tentatives échouées. Réessayez dans {LOGIN_ATTEMPT_WINDOW_MINUTES} minutes."

    return True, ""

def log_login_attempt(username: str, ip: str, success: bool):
    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    conn = get_db()
    conn.execute(
        "INSERT INTO login_attempts (id, username, ip, success, timestamp) VALUES (?,?,?,?,?)",
        (str(uuid.uuid4()), username, ip, 1 if success else 0, now)
    )
    conn.commit()
    conn.close()

def register_user(username: str, email: str, password: str, ip: str) -> Tuple[Optional[Dict], str]:
    username = username.strip()
    email = email.strip().lower()

    if not username or not email or not password:
        return None, "Tous les champs sont obligatoires."

    allowed, msg = _check_rate_limit(
        "registration_attempts", email, ip,
        MAX_REGISTRATION_ATTEMPTS,
        REGISTRATION_ATTEMPT_WINDOW_MINUTES
    )
    if not allowed:
        return None, msg

    if not validate_username(username):
        return None, "Nom d'utilisateur invalide (3-30 caractères, lettres/chiffres/-/_ uniquement)."

    if not validate_email(email):
        return None, "Adresse e-mail invalide."

    valid, msg = validate_password(password)
    if not valid:
        return None, msg

    conn = get_db()

    if get_user_by_username(username):
        _log_action("registration_attempts", email, ip, False)
        return None, f"Le nom d'utilisateur « {username} » est déjà pris."

    if get_user_by_email(email):
        _log_action("registration_attempts", email, ip, False)
        return None, "Cette adresse e-mail est déjà utilisée."

    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    uid = str(uuid.uuid4())
    password_hash = hash_password(password)
    verification_token = generate_token() if EMAIL_VERIFICATION_ENABLED else None
    is_verified = 0 if EMAIL_VERIFICATION_ENABLED else 1

    try:
        conn.execute(
            """INSERT INTO users
               (id, username, email, password_hash, ip, first_seen, last_seen,
                is_banned, is_verified, verification_token, created_at)
               VALUES (?,?,?,?,?,?,?,0,?,?,?)""",
            (uid, username, email, password_hash, ip, now, now, is_verified, verification_token, now)
        )
        conn.commit()

        if EMAIL_VERIFICATION_ENABLED and verification_token:
            sent, email_err = send_verification_email(email, verification_token)
            if not sent:
                conn.execute("DELETE FROM users WHERE id=?", (uid,))
                conn.commit()
                _log_action("registration_attempts", email, ip, False)
                logger.error(f"Verification email failed, rolled back user {email}: {email_err}")
                return None, "Impossible d'envoyer l'e-mail de vérification. Réessayez plus tard."

        _log_action("registration_attempts", email, ip, True)
        logger.info(f"New user registered: {username} ({email})")

        return {
            "id": uid,
            "username": username,
            "email": email,
            "needs_verification": EMAIL_VERIFICATION_ENABLED,
            "verification_token": verification_token if EMAIL_VERIFICATION_ENABLED else None
        }, ""

    except Exception as e:
        logger.error(f"Error registering user: {e}")
        _log_action("registration_attempts", email, ip, False)
        return None, "Erreur lors de la création du compte."
    finally:
        conn.close()

def verify_email(token: str) -> Tuple[bool, str]:
    conn = get_db()
    row = conn.execute(
        "SELECT id, email FROM users WHERE verification_token=? AND is_verified=0",
        (token,)
    ).fetchone()

    if not row:
        conn.close()
        return False, "Token de vérification invalide ou déjà utilisé."

    try:
        conn.execute(
            "UPDATE users SET is_verified=1, verification_token=NULL WHERE id=?",
            (row["id"],)
        )
        conn.commit()
        logger.info(f"Email verified for user: {row['email']}")
        return True, "Votre adresse e-mail a été vérifiée avec succès. Vous pouvez maintenant vous connecter."
    except Exception as e:
        logger.error(f"Error verifying email: {e}")
        return False, "Erreur lors de la vérification."
    finally:
        conn.close()


def resend_verification_email(email: str, ip: str) -> Tuple[bool, str]:
    email = email.strip().lower()

    if not validate_email(email):
        return False, "Adresse e-mail invalide."

    user = get_user_by_email(email)
    if not user:
        return False, "Aucun compte trouvé pour cette adresse e-mail."

    if user["is_verified"]:
        return False, "Ce compte est déjà vérifié."

    new_token = generate_token()
    sent, email_err = send_verification_email(email, new_token)
    if not sent:
        logger.error(f"SendGrid error: {email_err}")
        return False, "Impossible d'envoyer l'e-mail de vérification. Réessayez plus tard."

    conn = get_db()
    try:
        conn.execute(
            "UPDATE users SET verification_token=?, last_seen=? WHERE id=?",
            (new_token, datetime.now().isoformat(sep=' ', timespec='seconds'), user['id'])
        )
        conn.commit()
        logger.info(f"Verification email resent to {email}")
        return True, "L'e-mail de vérification a été renvoyé. Vérifiez votre boîte de réception."
    except Exception as e:
        logger.error(f"Error updating verification token for {email}: {e}")
        return False, "Une erreur est survenue lors du renvoi de l'e-mail."
    finally:
        conn.close()


def cleanup_unverified_accounts(hours: int = 24) -> int:
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat(sep=' ', timespec='seconds')
    conn = get_db()
    deleted = conn.execute(
        "DELETE FROM users WHERE is_verified=0 AND created_at < ?",
        (cutoff,)
    ).rowcount
    conn.commit()
    conn.close()
    if deleted > 0:
        logger.info(f"Deleted {deleted} unverified account(s) older than {hours} hours")
    return deleted


def login_user(username: str, password: str, ip: str) -> Tuple[Optional[Dict], str]:
    username = username.strip()

    if not username or not password:
        return None, "Remplissez tous les champs."

    allowed, msg = check_rate_limit(username, ip)
    if not allowed:
        logger.warning(f"Rate limit exceeded for {username} from {ip}")
        return None, msg

    user = get_user_by_username(username) or get_user_by_email(username.lower())

    if not user or not verify_password(password, user["password_hash"]):
        log_login_attempt(username, ip, False)
        logger.warning(f"Failed login attempt for {username} from {ip}")
        return None, "Nom d'utilisateur ou mot de passe incorrect."

    if user["is_banned"]:
        logger.warning(f"Banned user attempted login: {username}")
        return None, "Votre compte a été suspendu par un administrateur."

    if EMAIL_VERIFICATION_ENABLED and not user["is_verified"]:
        log_login_attempt(username, ip, False)
        logger.info(f"Unverified user attempted login: {username}")
        return None, "Veuillez vérifier votre adresse e-mail avant de vous connecter."

    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    conn = get_db()
    conn.execute("UPDATE users SET last_seen=?, ip=? WHERE id=?", (now, ip, user["id"]))
    conn.commit()
    conn.close()

    log_login_attempt(username, ip, True)
    logger.info(f"Successful login: {username} from {ip}")

    return {
        "id": user["id"],
        "username": user["username"],
        "email": user["email"]
    }, ""

def create_session(user_id: str, ip: str) -> str:
    now = datetime.now()
    expires = now + timedelta(hours=SESSION_TIMEOUT_HOURS)
    token = generate_token()
    token_hash = hash_token(token)

    conn = get_db()
    conn.execute(
        "INSERT INTO sessions (id, user_id, token, token_hash, created_at, expires_at, ip) VALUES (?,?,?,?,?,?,?)",
        (str(uuid.uuid4()), user_id, token, token_hash,
         now.isoformat(sep=" ", timespec="seconds"),
         expires.isoformat(sep=" ", timespec="seconds"), ip)
    )
    conn.commit()
    conn.close()

    return token

def validate_session(token: str) -> Optional[Dict]:
    if not token:
        return None

    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    token_hash = hash_token(token)
    conn = get_db()
    row = conn.execute(
        """SELECT s.user_id, u.username, u.email, u.is_banned
           FROM sessions s JOIN users u ON s.user_id = u.id
           WHERE (s.token_hash=? OR s.token=?) AND s.expires_at > ?""",
        (token_hash, token, now)
    ).fetchone()
    conn.close()

    if not row:
        return None

    if row["is_banned"]:
        return None

    return {
        "id": row["user_id"],
        "username": row["username"],
        "email": row["email"]
    }

def request_password_reset(email: str, ip: str) -> Tuple[bool, str]:
    email = email.strip().lower()

    if not validate_email(email):
        return False, "Adresse e-mail invalide."

    allowed, msg = _check_rate_limit(
        "password_reset_requests", email, ip,
        MAX_PASSWORD_RESET_REQUESTS,
        PASSWORD_RESET_WINDOW_MINUTES
    )
    if not allowed:
        return False, msg

    user = get_user_by_email(email)

    if not user or (EMAIL_VERIFICATION_ENABLED and not user['is_verified']):
        _log_action("password_reset_requests", email, ip, True)
        return True, "Si cette adresse e-mail existe, un lien de réinitialisation a été envoyé."

    token = generate_token()
    now = datetime.now().isoformat(sep=" ", timespec="seconds")

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO password_resets (id, user_id, token, created_at, used) VALUES (?,?,?,?,0)",
            (str(uuid.uuid4()), user["id"], token, now)
        )
        conn.commit()

        sent, email_err = send_password_reset_email(email, token)
        if not sent:
            conn.execute("DELETE FROM password_resets WHERE token=?", (token,))
            conn.commit()
            _log_action("password_reset_requests", email, ip, False)
            logger.error(f"Password reset email failed for {email}: {email_err}")
            return False, "Impossible d'envoyer l'e-mail de réinitialisation. Réessayez plus tard."

        _log_action("password_reset_requests", email, ip, True)
        logger.info(f"Password reset requested for: {email}")

        return True, "Si cette adresse e-mail existe, un lien de réinitialisation a été envoyé."
    except Exception as e:
        logger.error(f"Error creating password reset: {e}")
        _log_action("password_reset_requests", email, ip, False)
        return False, "Erreur lors de la demande de réinitialisation."
    finally:
        conn.close()

def reset_password(token: str, new_password: str) -> Tuple[bool, str]:
    valid, msg = validate_password(new_password)
    if not valid:
        return False, msg

    conn = get_db()
    row = conn.execute(
        "SELECT id, user_id, created_at FROM password_resets WHERE token=? AND used=0",
        (token,)
    ).fetchone()

    if not row:
        conn.close()
        return False, "Token de réinitialisation invalide ou déjà utilisé."

    if is_token_expired(row["created_at"], expiry_hours=1):
        conn.close()
        return False, "Ce lien de réinitialisation a expiré."

    try:
        password_hash = hash_password(new_password)
        conn.execute("UPDATE users SET password_hash=? WHERE id=?", (password_hash, row["user_id"]))
        conn.execute("UPDATE password_resets SET used=1 WHERE id=?", (row["id"],))
        conn.commit()
        logger.info(f"Password reset completed for user_id: {row['user_id']}")
        return True, "Votre mot de passe a été réinitialisé avec succès."
    except Exception as e:
        logger.error(f"Error resetting password: {e}")
        return False, "Erreur lors de la réinitialisation."
    finally:
        conn.close()

def update_user_ip(user_id: str, ip: str):
    conn = get_db()
    conn.execute("UPDATE users SET ip=? WHERE id=?", (ip, user_id))
    conn.commit()
    conn.close()
