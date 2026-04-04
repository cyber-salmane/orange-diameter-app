import logging
from pathlib import Path
from typing import Tuple
import pandas as pd

from db import get_db
from config import ADMIN_PASSWORD

logger = logging.getLogger(__name__)

def verify_admin_password(password: str) -> bool:
    return password == ADMIN_PASSWORD

def admin_get_users():
    conn = get_db()
    rows = conn.execute(
        """SELECT username, email, ip, first_seen, last_seen, is_banned, is_verified
           FROM users ORDER BY last_seen DESC"""
    ).fetchall()
    conn.close()

    if not rows:
        return pd.DataFrame(columns=["Utilisateur", "Email", "IP", "Inscription", "Dernière visite", "Vérifié", "Banni"])

    data = [{
        "Utilisateur": r["username"],
        "Email": r["email"],
        "IP": r["ip"] or "—",
        "Inscription": r["first_seen"],
        "Dernière visite": r["last_seen"],
        "Vérifié": "Oui" if r["is_verified"] else "Non",
        "Banni": "Oui" if r["is_banned"] else "Non",
    } for r in rows]
    return pd.DataFrame(data)

def admin_get_uploads():
    conn = get_db()
    rows = conn.execute(
        """SELECT u.username, u.ip, up.filename, up.upload_type, up.timestamp, up.path, up.file_size
           FROM uploads up JOIN users u ON u.id=up.user_id
           ORDER BY up.timestamp DESC LIMIT 100"""
    ).fetchall()
    conn.close()
    return rows

def admin_get_uploads_df():
    rows = admin_get_uploads()
    if not rows:
        return pd.DataFrame(columns=["Utilisateur", "IP", "Fichier", "Type", "Taille (KB)", "Date"])

    data = [{
        "Utilisateur": r["username"],
        "IP": r["ip"] or "—",
        "Fichier": r["filename"],
        "Type": r["upload_type"],
        "Taille (KB)": round(r["file_size"] / 1024, 1),
        "Date": r["timestamp"],
    } for r in rows]
    return pd.DataFrame(data)

def admin_get_images_list():
    rows = admin_get_uploads()
    import os
    return [r["path"] for r in rows if r["path"] and os.path.exists(r["path"])][:50]

def admin_delete_upload(filename: str) -> Tuple[str, pd.DataFrame, list]:
    filename = filename.strip()
    if not filename:
        return "Entrez un nom de fichier.", admin_get_uploads_df(), admin_get_images_list()

    conn = get_db()
    row = conn.execute("SELECT id, path FROM uploads WHERE filename=?", (filename,)).fetchone()

    if not row:
        conn.close()
        return f"Aucun fichier « {filename} » trouvé.", admin_get_uploads_df(), admin_get_images_list()

    try:
        path = Path(row["path"])
        if path.exists():
            path.unlink()
    except Exception as e:
        logger.error(f"Error deleting file {filename}: {e}")
        conn.close()
        return f"Erreur lors de la suppression du fichier: {e}", admin_get_uploads_df(), admin_get_images_list()

    conn.execute("DELETE FROM uploads WHERE id=?", (row["id"],))
    conn.commit()
    conn.close()
    logger.info(f"Admin deleted upload: {filename}")
    return f"Fichier « {filename} » supprimé avec succès.", admin_get_uploads_df(), admin_get_images_list()

def admin_delete_all_uploads() -> Tuple[str, pd.DataFrame, list]:
    conn = get_db()
    rows = conn.execute("SELECT path FROM uploads").fetchall()
    deleted = 0

    for r in rows:
        try:
            p = Path(r["path"])
            if p.exists():
                p.unlink()
                deleted += 1
        except Exception as e:
            logger.error(f"Error deleting file: {e}")

    conn.execute("DELETE FROM uploads")
    conn.commit()
    conn.close()
    logger.warning(f"Admin deleted all uploads: {deleted} files")
    return f"{deleted} image(s) supprimée(s).", admin_get_uploads_df(), admin_get_images_list()

def admin_get_stats():
    conn = get_db()
    total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    banned_users = conn.execute("SELECT COUNT(*) FROM users WHERE is_banned=1").fetchone()[0]
    verified_users = conn.execute("SELECT COUNT(*) FROM users WHERE is_verified=1").fetchone()[0]
    total_uploads = conn.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]
    total_analyses = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    total_oranges = conn.execute("SELECT COALESCE(SUM(num_oranges),0) FROM analyses").fetchone()[0]

    avg_row = conn.execute("SELECT AVG(avg_diameter) FROM analyses WHERE avg_diameter IS NOT NULL").fetchone()[0]
    avg_diam = round(avg_row, 2) if avg_row else 0

    from datetime import datetime
    today = datetime.now().date().isoformat()
    today_an = conn.execute("SELECT COUNT(*) FROM analyses WHERE timestamp LIKE ?", (today+"%",)).fetchone()[0]

    conn.close()

    return dict(
        total_users=total_users,
        banned_users=banned_users,
        verified_users=verified_users,
        total_uploads=total_uploads,
        total_analyses=total_analyses,
        total_oranges=total_oranges,
        avg_diam=avg_diam,
        today_analyses=today_an
    )

def admin_ban_user(identifier: str, ban: bool) -> str:
    identifier = identifier.strip()
    if not identifier:
        return "Entrez un nom d'utilisateur ou une adresse IP."

    conn = get_db()
    row = conn.execute("SELECT id, username FROM users WHERE username=? OR ip=?", (identifier, identifier)).fetchone()

    if not row:
        conn.close()
        return f"Aucun utilisateur trouvé pour « {identifier} »."

    conn.execute("UPDATE users SET is_banned=? WHERE id=?", (1 if ban else 0, row["id"]))
    conn.commit()
    conn.close()

    action = "banni" if ban else "débanni"
    logger.warning(f"Admin {action} user: {row['username']}")
    return f"Utilisateur « {row['username']} » {action} avec succès."

def admin_stats_html(s):
    return f"""
<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:16px;padding:8px 0">
  <div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:12px;padding:16px;text-align:center">
    <div style="font-size:2rem">👥</div>
    <div style="font-size:1.6rem;font-weight:700;color:#ea580c">{s['total_users']}</div>
    <div style="color:#9a3412;font-size:.85rem">Comptes créés</div>
  </div>
  <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:12px;padding:16px;text-align:center">
    <div style="font-size:2rem">✅</div>
    <div style="font-size:1.6rem;font-weight:700;color:#16a34a">{s['verified_users']}</div>
    <div style="color:#14532d;font-size:.85rem">Vérifiés</div>
  </div>
  <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:12px;padding:16px;text-align:center">
    <div style="font-size:2rem">🚫</div>
    <div style="font-size:1.6rem;font-weight:700;color:#dc2626">{s['banned_users']}</div>
    <div style="color:#991b1b;font-size:.85rem">Bannis</div>
  </div>
  <div style="background:#fffbeb;border:1px solid #fde68a;border-radius:12px;padding:16px;text-align:center">
    <div style="font-size:2rem">📤</div>
    <div style="font-size:1.6rem;font-weight:700;color:#d97706">{s['total_uploads']}</div>
    <div style="color:#92400e;font-size:.85rem">Images uploadées</div>
  </div>
  <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:12px;padding:16px;text-align:center">
    <div style="font-size:2rem">🔬</div>
    <div style="font-size:1.6rem;font-weight:700;color:#16a34a">{s['total_analyses']}</div>
    <div style="color:#14532d;font-size:.85rem">Analyses totales</div>
  </div>
  <div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:12px;padding:16px;text-align:center">
    <div style="font-size:2rem">🍊</div>
    <div style="font-size:1.6rem;font-weight:700;color:#ea580c">{s['total_oranges']}</div>
    <div style="color:#9a3412;font-size:.85rem">Oranges analysées</div>
  </div>
  <div style="background:#fdf4ff;border:1px solid #e9d5ff;border-radius:12px;padding:16px;text-align:center">
    <div style="font-size:2rem">📏</div>
    <div style="font-size:1.6rem;font-weight:700;color:#9333ea">{s['avg_diam']} mm</div>
    <div style="color:#581c87;font-size:.85rem">Diamètre moyen</div>
  </div>
  <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:12px;padding:16px;text-align:center">
    <div style="font-size:2rem">📅</div>
    <div style="font-size:1.6rem;font-weight:700;color:#2563eb">{s['today_analyses']}</div>
    <div style="color:#1e3a8a;font-size:.85rem">Aujourd'hui</div>
  </div>
</div>
"""
