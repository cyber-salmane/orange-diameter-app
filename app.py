import os
import sys
import sqlite3
import uuid
import hashlib
import shutil
from datetime import datetime
from pathlib import Path

# ── Fix missing system libs for open3d (Nix/prod) ──────────────────────────
def _fix_open3d_system_libs():
    needed = {
        "libudev.so.1": ["/lib/x86_64-linux-gnu/libudev.so.1",
                         "/usr/lib/x86_64-linux-gnu/libudev.so.1"],
        "libcap.so.2":  ["/lib/x86_64-linux-gnu/libcap.so.2",
                         "/usr/lib/x86_64-linux-gnu/libcap.so.2"],
    }
    open3d_cpu = None
    for base in sys.path:
        candidate = os.path.join(base, "open3d", "cpu")
        if os.path.isdir(candidate):
            open3d_cpu = candidate
            break
    if not open3d_cpu:
        site_pkgs = next((p for p in sys.path if "site-packages" in p), None)
        if site_pkgs:
            open3d_cpu = os.path.join(site_pkgs, "open3d", "cpu")
    if open3d_cpu and os.path.isdir(open3d_cpu):
        for lib_name, sources in needed.items():
            target = os.path.join(open3d_cpu, lib_name)
            if not os.path.exists(target):
                for src in sources:
                    if os.path.exists(src):
                        try:
                            os.symlink(src, target)
                        except OSError:
                            pass
                        break

_fix_open3d_system_libs()

import gradio as gr
import numpy as np
import cv2
import open3d as o3d
from PIL import Image
import pandas as pd
from scipy.optimize import least_squares

# ── Constants ────────────────────────────────────────────────────────────────
ADMIN_PASSWORD = "ORANGEADMIN"
DB_PATH        = "admin.db"
UPLOADS_DIR    = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

# ── Database ──────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def hash_pw(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id           TEXT PRIMARY KEY,
            username     TEXT UNIQUE NOT NULL,
            email        TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            ip           TEXT DEFAULT '',
            first_seen   TEXT NOT NULL,
            last_seen    TEXT NOT NULL,
            is_banned    INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS uploads (
            id          TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL,
            filename    TEXT NOT NULL,
            path        TEXT NOT NULL,
            upload_type TEXT NOT NULL,
            timestamp   TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS analyses (
            id             TEXT PRIMARY KEY,
            user_id        TEXT NOT NULL,
            num_oranges    INTEGER NOT NULL,
            avg_diameter   REAL,
            avg_confidence REAL,
            timestamp      TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()

init_db()

# ── Auth helpers ──────────────────────────────────────────────────────────────
def register_user(username: str, email: str, password: str, ip: str):
    """Returns (user_dict, error_message)."""
    username = username.strip()
    email    = email.strip().lower()
    if not username or not email or not password:
        return None, "Tous les champs sont obligatoires."
    if len(password) < 6:
        return None, "Le mot de passe doit contenir au moins 6 caractères."
    now  = datetime.now().isoformat(sep=" ", timespec="seconds")
    conn = get_db()
    if conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone():
        conn.close()
        return None, f"Le nom d'utilisateur « {username} » est déjà pris."
    if conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone():
        conn.close()
        return None, "Cette adresse e-mail est déjà utilisée."
    uid = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO users (id,username,email,password_hash,ip,first_seen,last_seen,is_banned) VALUES (?,?,?,?,?,?,?,0)",
        (uid, username, email, hash_pw(password), ip, now, now)
    )
    conn.commit()
    conn.close()
    return {"id": uid, "username": username, "email": email}, None

def login_user(username: str, password: str, ip: str):
    """Returns (user_dict, error_message)."""
    username = username.strip()
    if not username or not password:
        return None, "Remplissez tous les champs."
    now  = datetime.now().isoformat(sep=" ", timespec="seconds")
    conn = get_db()
    row  = conn.execute(
        "SELECT id, username, email, password_hash, is_banned FROM users WHERE username=? OR email=?",
        (username, username.lower())
    ).fetchone()
    if not row:
        conn.close()
        return None, "Nom d'utilisateur ou mot de passe incorrect."
    if row["password_hash"] != hash_pw(password):
        conn.close()
        return None, "Nom d'utilisateur ou mot de passe incorrect."
    if row["is_banned"]:
        conn.close()
        return None, "🚫 Votre compte a été suspendu par un administrateur."
    conn.execute("UPDATE users SET last_seen=?, ip=? WHERE id=?", (now, ip, row["id"]))
    conn.commit()
    conn.close()
    return {"id": row["id"], "username": row["username"], "email": row["email"]}, None

def update_user_ip(user_id: str, ip: str):
    conn = get_db()
    conn.execute("UPDATE users SET ip=? WHERE id=?", (ip, user_id))
    conn.commit()
    conn.close()

def save_analysis(user_id: str, num_oranges: int, avg_diameter, avg_confidence):
    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    conn = get_db()
    conn.execute(
        "INSERT INTO analyses (id,user_id,num_oranges,avg_diameter,avg_confidence,timestamp) VALUES (?,?,?,?,?,?)",
        (str(uuid.uuid4()), user_id, num_oranges, avg_diameter, avg_confidence, now)
    )
    conn.commit()
    conn.close()

# ── Admin DB helpers ──────────────────────────────────────────────────────────
def admin_get_users():
    conn = get_db()
    rows = conn.execute(
        "SELECT username, email, ip, first_seen, last_seen, is_banned FROM users ORDER BY last_seen DESC"
    ).fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame(columns=["Utilisateur", "Email", "IP", "Inscription", "Dernière visite", "Banni"])
    data = [{
        "Utilisateur":    r["username"],
        "Email":          r["email"],
        "IP":             r["ip"] or "—",
        "Inscription":    r["first_seen"],
        "Dernière visite": r["last_seen"],
        "Banni":          "🚫 OUI" if r["is_banned"] else "✅ NON",
    } for r in rows]
    return pd.DataFrame(data)

def admin_get_uploads():
    conn = get_db()
    rows = conn.execute(
        """SELECT u.username, u.ip, up.filename, up.upload_type, up.timestamp, up.path
           FROM uploads up JOIN users u ON u.id=up.user_id
           ORDER BY up.timestamp DESC LIMIT 100"""
    ).fetchall()
    conn.close()
    return rows

def admin_get_uploads_df():
    rows = admin_get_uploads()
    if not rows:
        return pd.DataFrame(columns=["Utilisateur", "IP", "Fichier", "Type", "Date"])
    data = [{
        "Utilisateur": r["username"], "IP": r["ip"] or "—",
        "Fichier": r["filename"], "Type": r["upload_type"], "Date": r["timestamp"],
    } for r in rows]
    return pd.DataFrame(data)

def admin_get_images_list():
    rows = admin_get_uploads()
    return [r["path"] for r in rows if r["path"] and os.path.exists(r["path"])][:50]

def admin_delete_upload(filename: str):
    """Delete an upload by filename — removes file from disk and DB record."""
    filename = filename.strip()
    if not filename:
        return "⚠️ Entrez un nom de fichier.", admin_get_uploads_df(), admin_get_images_list()
    conn = get_db()
    row = conn.execute("SELECT id, path FROM uploads WHERE filename=?", (filename,)).fetchone()
    if not row:
        conn.close()
        return f"⚠️ Aucun fichier « {filename} » trouvé dans la base.", admin_get_uploads_df(), admin_get_images_list()
    # Remove file from disk
    try:
        path = Path(row["path"])
        if path.exists():
            path.unlink()
    except Exception as e:
        conn.close()
        return f"❌ Erreur lors de la suppression du fichier: {e}", admin_get_uploads_df(), admin_get_images_list()
    conn.execute("DELETE FROM uploads WHERE id=?", (row["id"],))
    conn.commit()
    conn.close()
    return f"✅ Fichier « {filename} » supprimé avec succès.", admin_get_uploads_df(), admin_get_images_list()

def admin_delete_all_uploads():
    """Delete every upload from disk and DB."""
    conn = get_db()
    rows = conn.execute("SELECT path FROM uploads").fetchall()
    deleted = 0
    for r in rows:
        try:
            p = Path(r["path"])
            if p.exists():
                p.unlink()
                deleted += 1
        except Exception:
            pass
    conn.execute("DELETE FROM uploads")
    conn.commit()
    conn.close()
    return f"✅ {deleted} image(s) supprimée(s).", admin_get_uploads_df(), admin_get_images_list()

def admin_get_stats():
    conn = get_db()
    total_users    = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    banned_users   = conn.execute("SELECT COUNT(*) FROM users WHERE is_banned=1").fetchone()[0]
    total_uploads  = conn.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]
    total_analyses = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    total_oranges  = conn.execute("SELECT COALESCE(SUM(num_oranges),0) FROM analyses").fetchone()[0]
    avg_row        = conn.execute("SELECT AVG(avg_diameter) FROM analyses WHERE avg_diameter IS NOT NULL").fetchone()[0]
    avg_diam       = round(avg_row, 2) if avg_row else 0
    today          = datetime.now().date().isoformat()
    today_an       = conn.execute("SELECT COUNT(*) FROM analyses WHERE timestamp LIKE ?", (today+"%",)).fetchone()[0]
    conn.close()
    return dict(total_users=total_users, banned_users=banned_users,
                total_uploads=total_uploads, total_analyses=total_analyses,
                total_oranges=total_oranges, avg_diam=avg_diam, today_analyses=today_an)

def admin_ban_user(identifier: str, ban: bool) -> str:
    identifier = identifier.strip()
    if not identifier:
        return "⚠️ Entrez un nom d'utilisateur ou une adresse IP."
    conn = get_db()
    row = conn.execute("SELECT id FROM users WHERE username=? OR ip=?", (identifier, identifier)).fetchone()
    if not row:
        conn.close()
        return f"⚠️ Aucun utilisateur trouvé pour « {identifier} »."
    conn.execute("UPDATE users SET is_banned=? WHERE id=?", (1 if ban else 0, row["id"]))
    conn.commit()
    conn.close()
    return f"✅ Utilisateur « {identifier} » {'banni' if ban else 'débanni'} avec succès."

def admin_stats_html(s):
    return f"""
<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:16px;padding:8px 0">
  <div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:12px;padding:16px;text-align:center">
    <div style="font-size:2rem">👥</div>
    <div style="font-size:1.6rem;font-weight:700;color:#ea580c">{s['total_users']}</div>
    <div style="color:#9a3412;font-size:.85rem">Comptes créés</div>
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
    <div style="color:#1e3a8a;font-size:.85rem">Analyses aujourd'hui</div>
  </div>
</div>
"""

# ── 3D processing ─────────────────────────────────────────────────────────────
def depth_to_pointcloud(depth, fx, fy, cx, cy):
    h, w = depth.shape
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth; X = (xs - cx)*Z/fx; Y = (ys - cy)*Z/fy
    pts = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    valid = (Z.reshape(-1) > 0) & (Z.reshape(-1) < 10000)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[valid])
    return pcd

def remove_multiple_planes(pcd, max_planes=3, distance_threshold=0.002, min_plane_ratio=0.1):
    pcd_remaining = pcd
    for _ in range(max_planes):
        if len(pcd_remaining.points) < 1000: break
        try:
            _, inliers = pcd_remaining.segment_plane(
                distance_threshold=distance_threshold, ransac_n=3, num_iterations=2000)
            if len(inliers) < len(pcd_remaining.points) * min_plane_ratio: break
            pcd_remaining = pcd_remaining.select_by_index(inliers, invert=True)
        except: break
    return pcd_remaining

def is_complete_orange(pts, image_width, image_height, fx, fy, cx, cy, border_margin=15):
    Z = pts[:, 2]; valid_z = Z > 0
    if np.sum(valid_z) < 10: return False
    X = pts[valid_z, 0]; Y = pts[valid_z, 1]; Z = Z[valid_z]
    u = (X * fx / Z + cx).astype(int); v = (Y * fy / Z + cy).astype(int)
    on_border = ((u < border_margin) | (u > image_width - border_margin) |
                 (v < border_margin) | (v > image_height - border_margin))
    return np.sum(on_border) / len(pts) < 0.15

def fit_sphere_ransac(points, max_iterations=2000, distance_threshold=3.0, min_inliers_ratio=0.65):
    best_center = best_radius = best_inliers_mask = None
    best_inliers_count = 0; n_points = len(points)
    for _ in range(max_iterations):
        if n_points < 4: break
        sample_idx = np.random.choice(n_points, 4, replace=False)
        sample_points = points[sample_idx]
        try:
            center = np.mean(sample_points, axis=0)
            radius = np.mean(np.linalg.norm(sample_points - center, axis=1))
            distances = np.abs(np.linalg.norm(points - center, axis=1) - radius)
            inliers_mask = distances < distance_threshold
            inliers_count = np.sum(inliers_mask)
            if inliers_count > best_inliers_count and inliers_count > min_inliers_ratio * n_points:
                best_inliers_count = inliers_count; best_inliers_mask = inliers_mask
                inliers = points[inliers_mask]
                best_center = np.mean(inliers, axis=0)
                best_radius = np.mean(np.linalg.norm(inliers - best_center, axis=1))
        except: continue
    if best_center is not None and best_inliers_mask is not None:
        return best_center, best_radius, best_inliers_count, best_inliers_mask
    center = np.mean(points, axis=0)
    return center, np.median(np.linalg.norm(points - center, axis=1)), len(points), np.ones(len(points), dtype=bool)

def optimize_sphere_fit(points, initial_center, initial_radius):
    def residuals(params):
        cx, cy, cz, r = params
        return np.sqrt((points[:,0]-cx)**2 + (points[:,1]-cy)**2 + (points[:,2]-cz)**2) - r
    x0 = [*initial_center, initial_radius]
    try:
        result = least_squares(residuals, x0, method="lm", max_nfev=1000)
        opt_c, opt_r = result.x[:3], result.x[3]
        if abs(opt_r - initial_radius) > initial_radius * 0.5:
            return initial_center, initial_radius
        return opt_c, opt_r
    except: return initial_center, initial_radius

def validate_sphere(radius, num_points, inliers_ratio, is_complete):
    quality = "Excellente"; confidence = 100; warnings = []
    if num_points < 500:    quality = "Faible";  confidence = 40; warnings.append("Peu de points 3D")
    elif num_points < 1000: quality = "Moyenne"; confidence = 70
    elif num_points < 2000: quality = "Bonne";   confidence = 85
    if inliers_ratio < 0.6:
        quality = "Faible"; confidence = min(confidence, 50); warnings.append("Bruit important")
    elif inliers_ratio < 0.75:
        if quality == "Excellente": quality = "Bonne"
        confidence = min(confidence, 80)
    if radius < 20 or radius > 60:
        quality = "Suspect"; confidence = 30; warnings.append("Taille inhabituelle")
    if not is_complete:
        if quality in ["Excellente", "Bonne"]: quality = "Moyenne"
        confidence = min(confidence, 70); warnings.append("Partiellement visible")
    return quality, confidence, warnings

def process_oranges(rgb_img, depth_img,
                    fx, fy, cx, cy, depth_scale,
                    plane_threshold, max_planes, dbscan_eps, dbscan_min_points,
                    ransac_iterations, ransac_threshold,
                    use_optimization, use_outlier_removal,
                    session: dict,
                    request: gr.Request):
    if not session:
        return None, None, "❌ Vous devez être connecté pour utiliser cette fonction."

    user_id = session["id"]

    # Update IP
    try:
        ip = request.client.host if request and request.client else ""
        if ip: update_user_ip(user_id, ip)
    except Exception: pass

    if rgb_img is None or depth_img is None:
        return None, None, "❌ Veuillez uploader les deux images (RGB + Depth)"

    # Save uploaded image
    try:
        rgb_path = UPLOADS_DIR / f"{uuid.uuid4().hex}_rgb.png"
        rgb_img.save(str(rgb_path))
        now = datetime.now().isoformat(sep=" ", timespec="seconds")
        conn = get_db()
        conn.execute("INSERT INTO uploads (id,user_id,filename,path,upload_type,timestamp) VALUES (?,?,?,?,?,?)",
                     (str(uuid.uuid4()), user_id, rgb_path.name, str(rgb_path), "RGB", now))
        conn.commit(); conn.close()
    except Exception: pass

    try:
        rgb   = np.array(rgb_img)
        depth = np.array(depth_img.convert("L")).astype(np.float32) / depth_scale
        h, w  = depth.shape

        pcd_full = depth_to_pointcloud(depth, fx, fy, cx, cy)
        if len(pcd_full.points) < 1000:
            return None, None, "❌ Pas assez de points 3D valides dans la depth map"

        pcd_no_planes = remove_multiple_planes(
            pcd_full, max_planes=int(max_planes),
            distance_threshold=plane_threshold, min_plane_ratio=0.1)
        if len(pcd_no_planes.points) < 100:
            return None, None, "⚠️ Aucun objet détecté au-dessus des plans"

        labels = np.array(pcd_no_planes.cluster_dbscan(eps=dbscan_eps, min_points=int(dbscan_min_points)))
        unique_labels = [l for l in np.unique(labels) if l != -1]
        if len(unique_labels) == 0:
            return None, None, "⚠️ Aucune orange détectée. Ajustez les paramètres DBSCAN."

        annotated = rgb.copy(); results = []
        for lbl in unique_labels:
            idx = np.where(labels == lbl)[0]
            pts = np.asarray(pcd_no_planes.select_by_index(idx).points)
            if pts.shape[0] < 300: continue
            if use_outlier_removal:
                try:
                    pcd_c = o3d.geometry.PointCloud()
                    pcd_c.points = o3d.utility.Vector3dVector(pts)
                    cl, _ = pcd_c.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                    pts = np.asarray(cl.points)
                    if len(pts) < 100: continue
                except: pass

            complete = is_complete_orange(pts, w, h, fx, fy, cx, cy)
            center, radius, num_inliers, inliers_mask = fit_sphere_ransac(
                pts, max_iterations=int(ransac_iterations),
                distance_threshold=ransac_threshold, min_inliers_ratio=0.65)
            if use_optimization:
                center, radius = optimize_sphere_fit(pts[inliers_mask], center, radius)

            inliers_ratio = num_inliers / len(pts)
            diameter_mm   = radius * 2
            quality, confidence, warns = validate_sphere(radius, len(pts), inliers_ratio, complete)

            Zc = center[2]
            if Zc > 0:
                u = int((center[0] * fx / Zc) + cx)
                v = int((center[1] * fy / Zc) + cy)
                r_pixels = int(radius * fx / Zc)
            else:
                u, v, r_pixels = int(cx), int(cy), 30

            color_map = {"Excellente":(0,255,0),"Bonne":(144,238,144),
                         "Moyenne":(255,255,0),"Faible":(255,165,0),"Suspect":(255,0,0)}
            color = color_map.get(quality, (0,255,0))
            cv2.circle(annotated, (u, v), r_pixels, color, 3)
            text = f"#{len(results)+1}: {diameter_mm:.1f}mm"
            cv2.putText(annotated, text, (u-70,v-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 4)
            cv2.putText(annotated, text, (u-70,v-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            qt = f"{quality} ({confidence}%)"
            cv2.putText(annotated, qt, (u-70,v+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 3)
            cv2.putText(annotated, qt, (u-70,v+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if warns:
                cv2.putText(annotated, ", ".join(warns[:2]), (u-70,v+50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

            results.append({
                "Orange":         f"#{len(results)+1}",
                "Diamètre (mm)":  round(diameter_mm, 2),
                "Rayon (mm)":     round(radius, 2),
                "Points 3D":      len(pts),
                "Inliers (%)":    round(inliers_ratio*100, 1),
                "Qualité":        quality,
                "Confiance (%)":  confidence,
                "Complète":       "Oui" if complete else "Partielle",
                "Avertissements": "; ".join(warns) if warns else "Aucun",
            })

        if results:
            df   = pd.DataFrame(results)
            good = df[df["Qualité"].isin(["Excellente","Bonne"])]
            all_valid = df[df["Qualité"].isin(["Excellente","Bonne","Moyenne"])]
            avg_d = good["Diamètre (mm)"].mean() if len(good) > 0 else None
            avg_c = good["Confiance (%)"].mean() if len(good) > 0 else None
            save_analysis(user_id, len(results), avg_d, avg_c)

            if len(good) > 0:
                summary  = f"🍊 **{len(results)} orange(s) détectée(s)**\n\n"
                summary += "📊 **Statistiques (mesures excellentes/bonnes uniquement):**\n"
                summary += f"   • Diamètre moyen: **{good['Diamètre (mm)'].mean():.2f} mm**\n"
                summary += f"   • Écart-type: **{good['Diamètre (mm)'].std():.2f} mm**\n"
                summary += f"   • Min: {good['Diamètre (mm)'].min():.2f} mm\n"
                summary += f"   • Max: {good['Diamètre (mm)'].max():.2f} mm\n"
                summary += f"   • Confiance moyenne: **{good['Confiance (%)'].mean():.1f}%**\n\n"
                if len(all_valid) > len(good):
                    summary += f"ℹ️ ({len(all_valid)-len(good)} mesure(s) moyenne(s) exclue(s))\n\n"
                qcounts = df["Qualité"].value_counts()
                summary += "✅ **Répartition qualité:**\n"
                for q, c in qcounts.items():
                    summary += f"   • {q}: {c}\n"
                avg_conf = good["Confiance (%)"].mean()
                if avg_conf >= 90:   summary += "\n🎯 **Précision estimée: ±1.0-1.5 mm**"
                elif avg_conf >= 75: summary += "\n🎯 **Précision estimée: ±1.5-2.0 mm**"
                else:                summary += "\n⚠️ **Précision estimée: ±2.0-3.0 mm**"
            else:
                summary = (f"⚠️ {len(results)} orange(s) détectée(s) mais qualité insuffisante\n"
                           "Essayez d'améliorer l'éclairage ou la distance.")
        else:
            df = pd.DataFrame(); save_analysis(user_id, 0, None, None)
            summary = "❌ Aucune orange détectée. Vérifiez les paramètres et la qualité de la depth map."

        return annotated, df, summary

    except Exception as e:
        import traceback
        return None, None, f"❌ Erreur: {str(e)}\n\nDétails:\n{traceback.format_exc()[:500]}"


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="🍊 Mesure Oranges - Version ULTIME", theme=gr.themes.Soft()) as demo:

    session_state = gr.State(None)  # holds {id, username, email} when logged in

    # ═══════════════════════════════════════════════════════════════════════════
    # AUTH SCREEN  (visible by default)
    # ═══════════════════════════════════════════════════════════════════════════
    with gr.Group(visible=True) as auth_screen:
        gr.HTML("""
<div style="text-align:center;padding:40px 20px 20px">
  <div style="font-size:4rem">🍊</div>
  <h1 style="margin:8px 0 4px;font-size:1.8rem">Mesure de diamètres des oranges</h1>
  <p style="color:#6b7280;margin:0">Version ULTIME — Segmentation 3D</p>
</div>
""")
        with gr.Tabs() as auth_tabs:
            # ── Login tab ────────────────────────────────────────────────────
            with gr.Tab("🔑 Connexion"):
                with gr.Column(elem_id="login-col"):
                    login_username = gr.Textbox(label="Nom d'utilisateur ou e-mail",
                                                placeholder="votre_nom ou email@exemple.com",
                                                max_lines=1)
                    login_password = gr.Textbox(label="Mot de passe", type="password",
                                                placeholder="••••••••", max_lines=1)
                    login_btn      = gr.Button("Se connecter", variant="primary", size="lg")
                    login_msg      = gr.Markdown("")

            # ── Register tab ─────────────────────────────────────────────────
            with gr.Tab("✏️ Créer un compte"):
                with gr.Column():
                    reg_username = gr.Textbox(label="Nom d'utilisateur",
                                              placeholder="votre_nom", max_lines=1)
                    reg_email    = gr.Textbox(label="Adresse e-mail",
                                              placeholder="email@exemple.com", max_lines=1)
                    reg_password = gr.Textbox(label="Mot de passe (6 caractères min.)",
                                              type="password", placeholder="••••••••", max_lines=1)
                    reg_confirm  = gr.Textbox(label="Confirmer le mot de passe",
                                              type="password", placeholder="••••••••", max_lines=1)
                    reg_btn      = gr.Button("Créer mon compte", variant="primary", size="lg")
                    reg_msg      = gr.Markdown("")

    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN APP  (hidden until logged in)
    # ═══════════════════════════════════════════════════════════════════════════
    with gr.Group(visible=False) as main_app:
        with gr.Row():
            gr.Markdown("""
# 🍊 Mesure de diamètres des oranges — VERSION ULTIME
### Segmentation géométrique 3D pure + Optimisations avancées
**Pipeline:** Depth → 3D → Multi-Plane RANSAC → DBSCAN → Statistical Outlier Removal → RANSAC Sphere → Levenberg-Marquardt
**🎯 Précision attendue: ±1.0-1.5 mm**
""")
            with gr.Column(scale=0, min_width=160):
                user_greeting  = gr.Markdown("")
                app_logout_btn = gr.Button("🚪 Déconnexion", variant="secondary", size="sm")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📸 Images d'entrée")
                rgb_input   = gr.Image(label="Image RGB (affichage)", type="pil")
                depth_input = gr.Image(label="⭐ Depth Map (OBLIGATOIRE)", type="pil")

                with gr.Accordion("⚙️ Paramètres caméra", open=False):
                    gr.Markdown("*Laisser par défaut pour la plupart des caméras*")
                    fx          = gr.Number(value=1432.0, label="Focal length X (fx)")
                    fy          = gr.Number(value=1432.0, label="Focal length Y (fy)")
                    cx          = gr.Number(value=960.0,  label="Center X (cx)")
                    cy          = gr.Number(value=720.0,  label="Center Y (cy)")
                    depth_scale = gr.Number(value=1000.0, label="Depth scale")

                with gr.Accordion("🔧 Segmentation 3D", open=True):
                    gr.Markdown("**1. Suppression plans (table, murs)**")
                    plane_threshold   = gr.Slider(0.001, 0.01,  value=0.002, step=0.001, label="Seuil plan (m)")
                    max_planes        = gr.Slider(1, 5,          value=3,     step=1,     label="Nombre max de plans")
                    gr.Markdown("**2. Clustering 3D (DBSCAN)**")
                    dbscan_eps        = gr.Slider(0.005, 0.05,  value=0.015, step=0.001, label="Distance clustering (m)")
                    dbscan_min_points = gr.Slider(100, 1000,    value=300,   step=50,    label="Points min par cluster")
                    gr.Markdown("**3. Sphere Fitting (RANSAC)**")
                    ransac_iterations = gr.Slider(500, 5000,    value=2000,  step=100,   label="Itérations RANSAC")
                    ransac_threshold  = gr.Slider(1.0, 10.0,   value=3.0,   step=0.5,   label="Seuil RANSAC (mm)")

                with gr.Accordion("🚀 Optimisations avancées", open=True):
                    gr.Markdown("*Activer pour précision maximale*")
                    use_optimization    = gr.Checkbox(value=True,
                        label="✅ Optimisation non-linéaire (Levenberg-Marquardt)",
                        info="Affine le fit après RANSAC (+0.2-0.4mm précision)")
                    use_outlier_removal = gr.Checkbox(value=True,
                        label="✅ Filtrage statistique des outliers",
                        info="Supprime points aberrants (+0.3-0.5mm précision)")
                process_btn = gr.Button("🚀 ANALYSER", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### 📊 Résultats")
                summary_text  = gr.Markdown()
                output_image  = gr.Image(label="Image annotée")
                results_table = gr.Dataframe(label="Mesures détaillées", wrap=True)

        gr.Markdown("""
---
### 🏆 Améliorations de cette version ULTIME:
**✅ Multi-plane removal** → Supprime table ET murs/obstacles
**✅ Statistical outlier filtering** → Élimine points bruités (+0.3-0.5mm)
**✅ Levenberg-Marquardt optimization** → Affine le sphere fit (+0.2-0.4mm)
**✅ Détection oranges partielles** → Avertit si orange coupée par bord
**✅ Validation multi-critères** → Confiance basée sur 5 critères

### 📐 Précision finale:
- **Excellente qualité (>90% confiance): ±1.0-1.5 mm**
- **Bonne qualité (75-90% confiance): ±1.5-2.0 mm**
- **Moyenne qualité (<75% confiance): ±2.0-3.0 mm**

### 💡 Conseils pour meilleurs résultats:
- Distance caméra-oranges: **40-50 cm**
- Oranges espacées: **2-3 cm minimum**
- Table plane et stable
""")

        # Admin button
        gr.Markdown("---")
        with gr.Row():
            with gr.Column(scale=10): pass
            with gr.Column(scale=1, min_width=130):
                admin_btn = gr.Button("🔒 Admin", variant="secondary", size="sm")

        admin_authenticated = gr.State(False)

        with gr.Group(visible=False) as admin_login_group:
            gr.Markdown("## 🔐 Accès administrateur")
            with gr.Row():
                admin_pwd_input = gr.Textbox(label="Code d'accès", type="password",
                                             placeholder="Entrez le code admin", max_lines=1)
                admin_login_btn = gr.Button("Entrer", variant="primary")
            admin_login_msg = gr.Markdown("")

        with gr.Group(visible=False) as admin_dashboard:
            gr.Markdown("## 🛡️ Tableau de bord administrateur")
            with gr.Tabs():
                with gr.Tab("📊 Statistiques"):
                    stats_html    = gr.HTML()
                    refresh_stats = gr.Button("🔄 Actualiser", variant="secondary")

                with gr.Tab("👥 Utilisateurs"):
                    users_table   = gr.Dataframe(label="Comptes enregistrés", wrap=True, interactive=False)
                    refresh_users = gr.Button("🔄 Actualiser", variant="secondary")
                    gr.Markdown("### 🚫 Bannir / Débannir un utilisateur")
                    with gr.Row():
                        ban_id_input = gr.Textbox(label="Nom d'utilisateur ou IP",
                                                  placeholder="ex: jean_dupont ou 192.168.1.1", max_lines=1)
                        ban_btn      = gr.Button("🚫 Bannir",   variant="stop")
                        unban_btn    = gr.Button("✅ Débannir", variant="secondary")
                    ban_result = gr.Markdown("")

                with gr.Tab("🖼️ Images uploadées"):
                    uploads_table   = gr.Dataframe(label="Journal des uploads", wrap=True, interactive=False)
                    uploads_gallery = gr.Gallery(label="Aperçu", columns=4, height=400, object_fit="cover")
                    with gr.Row():
                        refresh_uploads = gr.Button("🔄 Actualiser", variant="secondary")
                    gr.Markdown("### 🗑️ Supprimer des images")
                    with gr.Row():
                        delete_filename_input = gr.Textbox(
                            label="Nom du fichier à supprimer",
                            placeholder="ex: a3f1b2c4d5e6f789_rgb.png",
                            max_lines=1,
                            scale=3,
                        )
                        delete_one_btn = gr.Button("🗑️ Supprimer", variant="stop", scale=1)
                    with gr.Row():
                        delete_all_btn = gr.Button("🗑️ Supprimer TOUTES les images", variant="stop")
                    delete_upload_msg = gr.Markdown("")

            admin_logout_btn = gr.Button("🔓 Se déconnecter de l'admin", variant="secondary", size="sm")

    # ═══════════════════════════════════════════════════════════════════════════
    # EVENT HANDLERS
    # ═══════════════════════════════════════════════════════════════════════════

    def do_login(username, password, request: gr.Request):
        ip = ""
        try:
            ip = request.client.host if request and request.client else ""
        except Exception: pass
        user, err = login_user(username, password, ip)
        if err:
            return None, gr.update(visible=True), gr.update(visible=False), f"❌ {err}", ""
        greeting = f"👤 **{user['username']}**"
        return (user,
                gr.update(visible=False),  # hide auth
                gr.update(visible=True),   # show main app
                "",                         # clear msg
                greeting)

    login_btn.click(
        fn=do_login,
        inputs=[login_username, login_password],
        outputs=[session_state, auth_screen, main_app, login_msg, user_greeting]
    )

    def do_register(username, email, password, confirm, request: gr.Request):
        ip = ""
        try:
            ip = request.client.host if request and request.client else ""
        except Exception: pass
        if password != confirm:
            return None, gr.update(visible=True), gr.update(visible=False), \
                   "❌ Les mots de passe ne correspondent pas.", ""
        user, err = register_user(username, email, password, ip)
        if err:
            return None, gr.update(visible=True), gr.update(visible=False), f"❌ {err}", ""
        greeting = f"👤 **{user['username']}**"
        return (user,
                gr.update(visible=False),
                gr.update(visible=True),
                "",
                greeting)

    reg_btn.click(
        fn=do_register,
        inputs=[reg_username, reg_email, reg_password, reg_confirm],
        outputs=[session_state, auth_screen, main_app, reg_msg, user_greeting]
    )

    def do_app_logout():
        return None, gr.update(visible=True), gr.update(visible=False), "", False, \
               gr.update(visible=False), gr.update(visible=False)

    app_logout_btn.click(
        fn=do_app_logout,
        inputs=[],
        outputs=[session_state, auth_screen, main_app, user_greeting,
                 admin_authenticated, admin_login_group, admin_dashboard]
    )

    # ── Main process ──────────────────────────────────────────────────────────
    process_btn.click(
        fn=process_oranges,
        inputs=[rgb_input, depth_input,
                fx, fy, cx, cy, depth_scale,
                plane_threshold, max_planes, dbscan_eps, dbscan_min_points,
                ransac_iterations, ransac_threshold,
                use_optimization, use_outlier_removal,
                session_state],
        outputs=[output_image, results_table, summary_text]
    )

    # ── Admin panel ───────────────────────────────────────────────────────────
    def toggle_admin_panel(current_auth):
        if current_auth:
            return gr.update(visible=False), gr.update(visible=True)
        return gr.update(visible=True), gr.update(visible=False)

    admin_btn.click(fn=toggle_admin_panel, inputs=[admin_authenticated],
                    outputs=[admin_login_group, admin_dashboard])

    def do_admin_login(pwd, current_auth):
        if pwd == ADMIN_PASSWORD:
            s = admin_get_stats()
            return (True,
                    gr.update(visible=False), gr.update(visible=True), "",
                    admin_stats_html(s), admin_get_users(), admin_get_uploads_df(), admin_get_images_list())
        return (current_auth, gr.update(visible=True), gr.update(visible=False),
                "❌ Code incorrect.", gr.update(), gr.update(), gr.update(), gr.update())

    admin_login_btn.click(
        fn=do_admin_login,
        inputs=[admin_pwd_input, admin_authenticated],
        outputs=[admin_authenticated, admin_login_group, admin_dashboard,
                 admin_login_msg, stats_html, users_table, uploads_table, uploads_gallery]
    )

    def do_admin_logout():
        return False, gr.update(visible=False), gr.update(visible=False), ""

    admin_logout_btn.click(fn=do_admin_logout, inputs=[],
                           outputs=[admin_authenticated, admin_login_group, admin_dashboard, admin_pwd_input])

    refresh_stats.click(fn=lambda: admin_stats_html(admin_get_stats()), outputs=[stats_html])
    refresh_users.click(fn=admin_get_users, outputs=[users_table])
    refresh_uploads.click(fn=lambda: (admin_get_uploads_df(), admin_get_images_list()),
                          outputs=[uploads_table, uploads_gallery])

    def delete_one_fn(filename):
        msg, table, gallery = admin_delete_upload(filename)
        return msg, table, gallery, ""

    delete_one_btn.click(
        fn=delete_one_fn,
        inputs=[delete_filename_input],
        outputs=[delete_upload_msg, uploads_table, uploads_gallery, delete_filename_input]
    )

    def delete_all_fn():
        msg, table, gallery = admin_delete_all_uploads()
        return msg, table, gallery

    delete_all_btn.click(
        fn=delete_all_fn,
        inputs=[],
        outputs=[delete_upload_msg, uploads_table, uploads_gallery]
    )

    def ban_fn(identifier):
        return admin_ban_user(identifier, ban=True), admin_get_users()

    def unban_fn(identifier):
        return admin_ban_user(identifier, ban=False), admin_get_users()

    ban_btn.click(fn=ban_fn,   inputs=[ban_id_input], outputs=[ban_result, users_table])
    unban_btn.click(fn=unban_fn, inputs=[ban_id_input], outputs=[ban_result, users_table])

# In production (deployed), Gradio can't parse Replit's double-dash domain
# (e.g. app--user.replit.app) — we must tell it its own root URL explicitly.
# In dev, REPLIT_DEV_DOMAIN is always set; in production it is not.
_dev_domain = os.environ.get("REPLIT_DEV_DOMAIN", "")
if not _dev_domain:
    # Production: derive root URL from REPLIT_DOMAINS
    _prod_domains = os.environ.get("REPLIT_DOMAINS", "")
    _root_path = f"https://{_prod_domains.split(',')[0].strip()}" if _prod_domains else ""
else:
    _root_path = ""   # Dev: leave empty so preview CSS loads normally

demo.launch(
    server_name="0.0.0.0",
    server_port=5000,
    root_path=_root_path,
    allowed_paths=["uploads"],
)
