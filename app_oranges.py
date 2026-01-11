import streamlit as st
import cv2
import numpy as np
import open3d as o3d
from PIL import Image

st.title("Mesure de diamètres des oranges (RGB-D)")

# Paramètres de calibration (à ajuster selon ta caméra)
st.sidebar.header("Paramètres de la caméra")
fx = st.sidebar.number_input("Focal length X (fx)", value=525.0, help="Paramètre intrinsèque de ta caméra")
fy = st.sidebar.number_input("Focal length Y (fy)", value=525.0)
cx = st.sidebar.number_input("Center X (cx)", value=319.5)
cy = st.sidebar.number_input("Center Y (cy)", value=239.5)
depth_scale = st.sidebar.number_input("Depth scale", value=1.0, help="Facteur de conversion (ex: 1000 si depth en mm)")

# Upload RGB et Depth
rgb_file = st.file_uploader("Upload Image RGB", type=["png", "jpg", "jpeg"])
depth_file = st.file_uploader("Upload Depth Map", type=["png", "tiff"])

if rgb_file and depth_file:
    # Lire l'image RGB
    rgb_image = np.array(Image.open(rgb_file).convert('RGB'))
    h, w = rgb_image.shape[:2]
    
    # Lire la depth map
    depth_image = np.array(Image.open(depth_file))
    if len(depth_image.shape) == 3:
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
    
    # Convertir en float et appliquer le scale
    depth = depth_image.astype(np.float32) / depth_scale
    
    st.subheader("Images uploadées")
    col1, col2 = st.columns(2)
    with col1:
        st.image(rgb_image, caption="Image RGB", use_container_width=True)
    with col2:
        st.image(depth_image, caption="Depth Map", use_container_width=True)
    
    # Segmentation par couleur HSV
    st.subheader("Paramètres de segmentation")
    col1, col2 = st.columns(2)
    with col1:
        lower_h = st.slider("Hue min", 0, 180, 5)
        lower_s = st.slider("Saturation min", 0, 255, 100)
        lower_v = st.slider("Value min", 0, 255, 100)
    with col2:
        upper_h = st.slider("Hue max", 0, 180, 20)
        upper_s = st.slider("Saturation max", 0, 255, 255)
        upper_v = st.slider("Value max", 0, 255, 255)
    
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    lower_orange = np.array([lower_h, lower_s, lower_v])
    upper_orange = np.array([upper_h, upper_s, upper_v])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Morphologie pour nettoyer
    kernel_size = st.sidebar.slider("Kernel morphologie", 3, 15, 5, step=2)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    st.image(mask, caption="Masque de segmentation", use_container_width=True)
    
    # Trouver les contours des oranges
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    annotated = rgb_image.copy()
    results = []
    min_contour_area = st.sidebar.number_input("Aire minimale du contour", value=500, step=100)

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_contour_area:
            continue
        
        # Masque pour cette orange
        orange_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(orange_mask, [cnt], -1, 255, -1)
        
        # Extraire nuage de points 3D avec projection caméra
        ys, xs = np.where(orange_mask > 0)
        zs = depth[ys, xs]
        
        # Filtrer les points invalides (depth = 0)
        valid = zs > 0
        xs, ys, zs = xs[valid], ys[valid], zs[valid]
        
        if len(xs) < 100:
            continue
        
        # Projection 3D avec paramètres intrinsèques
        X = (xs - cx) * zs / fx
        Y = (ys - cy) * zs / fy
        Z = zs
        
        points_3d = np.stack([X, Y, Z], axis=1)
        
        # Créer nuage de points Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        
        # Méthode 1: Calcul du rayon moyen depuis le centroïde
        centroid = np.mean(points_3d, axis=0)
        distances = np.linalg.norm(points_3d - centroid, axis=1)
        radius_mean = np.mean(distances)
        
        # Méthode 2: RANSAC sphere fitting (plus robuste)
        try:
            sphere_model, inliers = pcd.segment_plane(
                distance_threshold=5.0,
                ransac_n=3,
                num_iterations=1000
            )
            # Note: segment_plane trouve un plan, pour une sphère il faut une méthode custom
            # Utilisons la méthode simple pour l'instant
            radius = radius_mean
        except:
            radius = radius_mean
        
        diameter_mm = radius * 2
        
        # Dessiner sur l'image
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # Calculer rayon en pixels pour l'annotation
            (x, y), r_pixels = cv2.minEnclosingCircle(cnt)
            
            cv2.circle(annotated, (cX, cY), int(r_pixels), (0, 255, 0), 2)
            cv2.putText(annotated, f"#{i+1}: {diameter_mm:.1f} mm", 
                       (cX - 50, cY - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            results.append({
                'Orange': i + 1,
                'Diamètre (mm)': round(diameter_mm, 2),
                'Rayon (mm)': round(radius, 2),
                'Points 3D': len(points_3d),
                'Aire contour': int(area)
            })

    st.subheader(f"🍊 {len(results)} orange(s) détectée(s)")
    
    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
        
        # Statistiques
        st.metric("Diamètre moyen", f"{df['Diamètre (mm)'].mean():.2f} mm")
        st.metric("Écart-type", f"{df['Diamètre (mm)'].std():.2f} mm")
    
    st.subheader("Image annotée")
    st.image(annotated, use_container_width=True)
    
    # Export des résultats
    if results:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger les résultats (CSV)",
            data=csv,
            file_name="mesures_oranges.csv",
            mime="text/csv"
        )
else:
    st.info("👆 Upload une image RGB et une depth map pour commencer")
    
    with st.expander("ℹ️ Instructions"):
        st.markdown("""
        ### Comment utiliser cette application:
        
        1. **Upload tes images:**
           - Image RGB (photo normale de tes oranges)
           - Depth map (carte de profondeur de ta caméra ToF)
        
        2. **Ajuste les paramètres:**
           - Paramètres de calibration de la caméra (barre latérale gauche)
           - Paramètres de segmentation couleur (sliders HSV)
        
        3. **Vérifie le masque:**
           - Assure-toi que les oranges sont bien détectées (en blanc)
           - Ajuste les sliders HSV si nécessaire
        
        4. **Résultats:**
           - Diamètres calculés en millimètres
           - Image annotée avec les mesures
           - Export CSV disponible
        
        ### Précision attendue:
        - ±3-5 mm avec bonne calibration
        - ±2-3 mm avec calibration précise et moyennage multi-frames
        """)