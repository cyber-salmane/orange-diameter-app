import os
import sys
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image

import numpy as np
import cv2
import open3d as o3d
import pandas as pd
from scipy.optimize import least_squares

from config import UPLOADS_DIR
from db import get_db
from utils import validate_file, sanitize_filename

logger = logging.getLogger(__name__)

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

def depth_to_pointcloud(depth, fx, fy, cx, cy):
    h, w = depth.shape
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth
    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy
    pts = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    valid = (Z.reshape(-1) > 0) & (Z.reshape(-1) < 10000)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[valid])
    return pcd

def remove_multiple_planes(pcd, max_planes=3, distance_threshold=0.002, min_plane_ratio=0.1):
    pcd_remaining = pcd
    for _ in range(max_planes):
        if len(pcd_remaining.points) < 1000:
            break
        try:
            _, inliers = pcd_remaining.segment_plane(
                distance_threshold=distance_threshold, ransac_n=3, num_iterations=2000)
            if len(inliers) < len(pcd_remaining.points) * min_plane_ratio:
                break
            pcd_remaining = pcd_remaining.select_by_index(inliers, invert=True)
        except:
            break
    return pcd_remaining

def is_complete_orange(pts, image_width, image_height, fx, fy, cx, cy, border_margin=15):
    Z = pts[:, 2]
    valid_z = Z > 0
    if np.sum(valid_z) < 10:
        return False
    X = pts[valid_z, 0]
    Y = pts[valid_z, 1]
    Z = Z[valid_z]
    u = (X * fx / Z + cx).astype(int)
    v = (Y * fy / Z + cy).astype(int)
    on_border = ((u < border_margin) | (u > image_width - border_margin) |
                 (v < border_margin) | (v > image_height - border_margin))
    return np.sum(on_border) / len(pts) < 0.15

def fit_sphere_ransac(points, max_iterations=2000, distance_threshold=3.0, min_inliers_ratio=0.65):
    best_center = best_radius = best_inliers_mask = None
    best_inliers_count = 0
    n_points = len(points)

    for _ in range(max_iterations):
        if n_points < 4:
            break
        sample_idx = np.random.choice(n_points, 4, replace=False)
        sample_points = points[sample_idx]
        try:
            center = np.mean(sample_points, axis=0)
            radius = np.mean(np.linalg.norm(sample_points - center, axis=1))
            distances = np.abs(np.linalg.norm(points - center, axis=1) - radius)
            inliers_mask = distances < distance_threshold
            inliers_count = np.sum(inliers_mask)
            if inliers_count > best_inliers_count and inliers_count > min_inliers_ratio * n_points:
                best_inliers_count = inliers_count
                best_inliers_mask = inliers_mask
                inliers = points[inliers_mask]
                best_center = np.mean(inliers, axis=0)
                best_radius = np.mean(np.linalg.norm(inliers - best_center, axis=1))
        except:
            continue

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
    except:
        return initial_center, initial_radius

def validate_sphere(radius, num_points, inliers_ratio, is_complete):
    quality = "Excellente"
    confidence = 100
    warnings = []

    if num_points < 500:
        quality = "Faible"
        confidence = 40
        warnings.append("Peu de points 3D")
    elif num_points < 1000:
        quality = "Moyenne"
        confidence = 70
    elif num_points < 2000:
        quality = "Bonne"
        confidence = 85

    if inliers_ratio < 0.6:
        quality = "Faible"
        confidence = min(confidence, 50)
        warnings.append("Bruit important")
    elif inliers_ratio < 0.75:
        if quality == "Excellente":
            quality = "Bonne"
        confidence = min(confidence, 80)

    if radius < 20 or radius > 60:
        quality = "Suspect"
        confidence = 30
        warnings.append("Taille inhabituelle")

    if not is_complete:
        if quality in ["Excellente", "Bonne"]:
            quality = "Moyenne"
        confidence = min(confidence, 70)
        warnings.append("Partiellement visible")

    return quality, confidence, warnings

def save_upload(user_id: str, filename: str, path: str, upload_type: str, file_size: int):
    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO uploads (id, user_id, filename, path, upload_type, file_size, timestamp) VALUES (?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), user_id, filename, path, upload_type, file_size, now)
        )
        conn.commit()
    except Exception as e:
        logger.error(f"Error saving upload: {e}")
    finally:
        conn.close()

def save_analysis(user_id: str, num_oranges: int, avg_diameter, avg_confidence):
    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    conn = get_db()
    try:
        analysis_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO analyses (id, user_id, num_oranges, avg_diameter, avg_confidence, timestamp) VALUES (?,?,?,?,?,?)",
            (analysis_id, user_id, num_oranges, avg_diameter, avg_confidence, now)
        )
        conn.commit()
        return analysis_id
    except Exception as e:
        logger.error(f"Error saving analysis: {e}")
        return None
    finally:
        conn.close()

def process_oranges(
    rgb_img, depth_img,
    fx, fy, cx, cy, depth_scale,
    plane_threshold, max_planes, dbscan_eps, dbscan_min_points,
    ransac_iterations, ransac_threshold,
    use_optimization, use_outlier_removal,
    user_id: str
) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame], str]:

    if not user_id:
        return None, None, "Vous devez être connecté pour utiliser cette fonction."

    if rgb_img is None or depth_img is None:
        return None, None, "Veuillez uploader les deux images (RGB + Depth)"

    try:
        rgb_path = UPLOADS_DIR / f"{uuid.uuid4().hex}_rgb.png"
        rgb_img.save(str(rgb_path))
        file_size = rgb_path.stat().st_size
        save_upload(user_id, rgb_path.name, str(rgb_path), "RGB", file_size)
        logger.info(f"User {user_id} uploaded image: {rgb_path.name}")
    except Exception as e:
        logger.error(f"Error saving upload: {e}")

    try:
        rgb = np.array(rgb_img)
        depth = np.array(depth_img.convert("L")).astype(np.float32) / depth_scale
        h, w = depth.shape

        pcd_full = depth_to_pointcloud(depth, fx, fy, cx, cy)
        if len(pcd_full.points) < 1000:
            return None, None, "Pas assez de points 3D valides dans la depth map"

        pcd_no_planes = remove_multiple_planes(
            pcd_full, max_planes=int(max_planes),
            distance_threshold=plane_threshold, min_plane_ratio=0.1)

        if len(pcd_no_planes.points) < 100:
            return None, None, "Aucun objet détecté au-dessus des plans"

        labels = np.array(pcd_no_planes.cluster_dbscan(eps=dbscan_eps, min_points=int(dbscan_min_points)))
        unique_labels = [l for l in np.unique(labels) if l != -1]

        if len(unique_labels) == 0:
            return None, None, "Aucune orange détectée. Ajustez les paramètres DBSCAN."

        annotated = rgb.copy()
        results = []

        for lbl in unique_labels:
            idx = np.where(labels == lbl)[0]
            pts = np.asarray(pcd_no_planes.select_by_index(idx).points)
            if pts.shape[0] < 300:
                continue

            if use_outlier_removal:
                try:
                    pcd_c = o3d.geometry.PointCloud()
                    pcd_c.points = o3d.utility.Vector3dVector(pts)
                    cl, _ = pcd_c.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                    pts = np.asarray(cl.points)
                    if len(pts) < 100:
                        continue
                except:
                    pass

            complete = is_complete_orange(pts, w, h, fx, fy, cx, cy)
            center, radius, num_inliers, inliers_mask = fit_sphere_ransac(
                pts, max_iterations=int(ransac_iterations),
                distance_threshold=ransac_threshold, min_inliers_ratio=0.65)

            if use_optimization:
                center, radius = optimize_sphere_fit(pts[inliers_mask], center, radius)

            inliers_ratio = num_inliers / len(pts)
            diameter_mm = radius * 2
            quality, confidence, warns = validate_sphere(radius, len(pts), inliers_ratio, complete)

            Zc = center[2]
            if Zc > 0:
                u = int((center[0] * fx / Zc) + cx)
                v = int((center[1] * fy / Zc) + cy)
                r_pixels = int(radius * fx / Zc)
            else:
                u, v, r_pixels = int(cx), int(cy), 30

            color_map = {
                "Excellente": (0, 255, 0),
                "Bonne": (144, 238, 144),
                "Moyenne": (255, 255, 0),
                "Faible": (255, 165, 0),
                "Suspect": (255, 0, 0)
            }
            color = color_map.get(quality, (0, 255, 0))

            cv2.circle(annotated, (u, v), r_pixels, color, 3)
            text = f"#{len(results)+1}: {diameter_mm:.1f}mm"
            cv2.putText(annotated, text, (u-70, v-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4)
            cv2.putText(annotated, text, (u-70, v-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            qt = f"{quality} ({confidence}%)"
            cv2.putText(annotated, qt, (u-70, v+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)
            cv2.putText(annotated, qt, (u-70, v+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if warns:
                cv2.putText(annotated, ", ".join(warns[:2]), (u-70, v+50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            results.append({
                "Orange": f"#{len(results)+1}",
                "Diamètre (mm)": round(diameter_mm, 2),
                "Rayon (mm)": round(radius, 2),
                "Points 3D": len(pts),
                "Inliers (%)": round(inliers_ratio*100, 1),
                "Qualité": quality,
                "Confiance (%)": confidence,
                "Complète": "Oui" if complete else "Partielle",
                "Avertissements": "; ".join(warns) if warns else "Aucun",
            })

        if results:
            df = pd.DataFrame(results)
            good = df[df["Qualité"].isin(["Excellente", "Bonne"])]
            avg_d = good["Diamètre (mm)"].mean() if len(good) > 0 else None
            avg_c = good["Confiance (%)"].mean() if len(good) > 0 else None
            save_analysis(user_id, len(results), avg_d, avg_c)
            logger.info(f"Analysis completed for user {user_id}: {len(results)} oranges detected")

            if len(good) > 0:
                summary = f"**{len(results)} orange(s) détectée(s)**\n\n"
                summary += "**Statistiques (mesures excellentes/bonnes uniquement):**\n"
                summary += f"   - Diamètre moyen: **{good['Diamètre (mm)'].mean():.2f} mm**\n"
                summary += f"   - Écart-type: **{good['Diamètre (mm)'].std():.2f} mm**\n"
                summary += f"   - Min: {good['Diamètre (mm)'].min():.2f} mm\n"
                summary += f"   - Max: {good['Diamètre (mm)'].max():.2f} mm\n"
                summary += f"   - Confiance moyenne: **{good['Confiance (%)'].mean():.1f}%**\n\n"
                if len(df) > len(good):
                    summary += f"({len(df)-len(good)} mesure(s) de qualité moyenne/faible)\n\n"
                qcounts = df["Qualité"].value_counts()
                summary += "**Répartition qualité:**\n"
                for q, c in qcounts.items():
                    summary += f"   - {q}: {c}\n"
                avg_conf = good["Confiance (%)"].mean()
                if avg_conf >= 90:
                    summary += "\n**Précision estimée: ±1.0-1.5 mm**"
                elif avg_conf >= 75:
                    summary += "\n**Précision estimée: ±1.5-2.0 mm**"
                else:
                    summary += "\n**Précision estimée: ±2.0-3.0 mm**"
            else:
                summary = (f"{len(results)} orange(s) détectée(s) mais qualité insuffisante\n"
                           "Essayez d'améliorer l'éclairage ou la distance.")
        else:
            df = pd.DataFrame()
            save_analysis(user_id, 0, None, None)
            summary = "Aucune orange détectée. Vérifiez les paramètres et la qualité de la depth map."

        return annotated, df, summary

    except Exception as e:
        logger.error(f"Error processing oranges: {e}")
        import traceback
        return None, None, f"Erreur: {str(e)}\n\nDétails:\n{traceback.format_exc()[:500]}"
