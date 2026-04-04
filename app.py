import os
import logging
from pathlib import Path
from datetime import datetime

import gradio as gr
import pandas as pd

from config import UPLOADS_DIR, CLEANUP_OLD_FILES_DAYS, EMAIL_VERIFICATION_ENABLED
from db import (
    init_db, cleanup_expired_sessions, cleanup_old_login_attempts,
    get_user_analyses, get_user_stats
)
from auth import (
    register_user, login_user, verify_email, request_password_reset,
    reset_password, create_session, validate_session, update_user_ip
)
from processing import process_oranges
from admin import (
    verify_admin_password, admin_get_users, admin_get_uploads_df,
    admin_get_images_list, admin_delete_upload, admin_delete_all_uploads,
    admin_get_stats, admin_ban_user, admin_stats_html
)
from utils import cleanup_old_files

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

init_db()
cleanup_expired_sessions()
cleanup_old_login_attempts()
cleanup_old_files(UPLOADS_DIR, CLEANUP_OLD_FILES_DAYS)

custom_css = """
.auth-container {
    max-width: 500px;
    margin: 0 auto;
    padding: 2rem;
}

.main-header {
    text-align: center;
    padding: 2rem 1rem 1rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 12px;
    margin-bottom: 2rem;
}

.stat-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    transition: transform 0.2s, box-shadow 0.2s;
}

.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.success-message {
    background: #d1fae5;
    border: 1px solid #6ee7b7;
    color: #065f46;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.error-message {
    background: #fee2e2;
    border: 1px solid #fca5a5;
    color: #991b1b;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.warning-message {
    background: #fef3c7;
    border: 1px solid #fcd34d;
    color: #92400e;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.info-box {
    background: #dbeafe;
    border: 1px solid #93c5fd;
    color: #1e3a8a;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.btn-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 0.75rem 2rem;
    border-radius: 8px;
    font-weight: 600;
    cursor: pointer;
    transition: transform 0.2s;
}

.btn-primary:hover {
    transform: translateY(-1px);
}

.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e5e7eb;
}

.dashboard-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

footer {
    text-align: center;
    padding: 2rem;
    color: #6b7280;
    font-size: 0.875rem;
}
"""

with gr.Blocks(title="Mesure Oranges - Production", theme=gr.themes.Soft(), css=custom_css) as demo:

    session_token = gr.State(None)

    with gr.Group(visible=True, elem_classes="auth-container") as auth_screen:
        gr.HTML("""
<div style="text-align:center;padding:40px 20px 20px">
  <div style="font-size:4rem">🍊</div>
  <h1 style="margin:8px 0 4px;font-size:2rem;color:#1f2937">Mesure de diamètres des oranges</h1>
  <p style="color:#6b7280;margin:0">Analyse 3D de précision professionnelle</p>
</div>
""")

        with gr.Tabs() as auth_tabs:
            with gr.Tab("Connexion"):
                gr.Markdown("### Connectez-vous à votre compte")
                login_username = gr.Textbox(label="Nom d'utilisateur ou e-mail", placeholder="votre_nom", max_lines=1)
                login_password = gr.Textbox(label="Mot de passe", type="password", placeholder="••••••••", max_lines=1)
                login_btn = gr.Button("Se connecter", variant="primary", size="lg")
                login_msg = gr.Markdown("")
                forgot_password_btn = gr.Button("Mot de passe oublié ?", variant="secondary", size="sm")

            with gr.Tab("Créer un compte"):
                gr.Markdown("### Créez votre compte gratuit")
                reg_username = gr.Textbox(label="Nom d'utilisateur", placeholder="votre_nom", max_lines=1)
                reg_email = gr.Textbox(label="Adresse e-mail", placeholder="email@exemple.com", max_lines=1)
                reg_password = gr.Textbox(label="Mot de passe", type="password", placeholder="••••••••", max_lines=1)
                reg_confirm = gr.Textbox(label="Confirmer le mot de passe", type="password", placeholder="••••••••", max_lines=1)
                if EMAIL_VERIFICATION_ENABLED:
                    gr.Markdown("**Note:** Un e-mail de vérification sera envoyé à votre adresse.")
                reg_btn = gr.Button("Créer mon compte", variant="primary", size="lg")
                reg_msg = gr.Markdown("")

            with gr.Tab("Réinitialiser mot de passe"):
                gr.Markdown("### Réinitialisez votre mot de passe")
                reset_email = gr.Textbox(label="Adresse e-mail", placeholder="email@exemple.com", max_lines=1)
                reset_request_btn = gr.Button("Envoyer le lien de réinitialisation", variant="primary")
                reset_request_msg = gr.Markdown("")

                gr.Markdown("---")
                gr.Markdown("### Vous avez reçu un code de réinitialisation ?")
                reset_token = gr.Textbox(label="Code de réinitialisation", placeholder="Collez le code ici", max_lines=1)
                reset_new_password = gr.Textbox(label="Nouveau mot de passe", type="password", placeholder="••••••••", max_lines=1)
                reset_confirm_btn = gr.Button("Réinitialiser le mot de passe", variant="primary")
                reset_confirm_msg = gr.Markdown("")

    with gr.Group(visible=False) as main_app:
        with gr.Row():
            gr.HTML("""
<div class="main-header">
  <h1 style="margin:0 0 0.5rem;font-size:2rem">🍊 Mesure de diamètres des oranges</h1>
  <p style="margin:0;opacity:0.9">Segmentation géométrique 3D pure + Optimisations avancées</p>
  <p style="margin:0.5rem 0 0;font-size:0.9rem;opacity:0.8">Précision attendue: ±1.0-1.5 mm</p>
</div>
""")

        with gr.Row():
            with gr.Column(scale=5):
                user_greeting = gr.Markdown("")
            with gr.Column(scale=1, min_width=200):
                gr.Row()
                with gr.Row():
                    dashboard_btn = gr.Button("📊 Mon tableau de bord", variant="secondary", size="sm")
                    app_logout_btn = gr.Button("🚪 Déconnexion", variant="secondary", size="sm")

        with gr.Tabs() as main_tabs:
            with gr.Tab("🔬 Analyser"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📸 Images d'entrée")
                        rgb_input = gr.Image(label="Image RGB (affichage)", type="pil")
                        depth_input = gr.Image(label="Depth Map (OBLIGATOIRE)", type="pil")

                        with gr.Accordion("⚙️ Paramètres caméra", open=False):
                            gr.Markdown("*Laisser par défaut pour la plupart des caméras*")
                            fx = gr.Number(value=1432.0, label="Focal length X (fx)")
                            fy = gr.Number(value=1432.0, label="Focal length Y (fy)")
                            cx = gr.Number(value=960.0, label="Center X (cx)")
                            cy = gr.Number(value=720.0, label="Center Y (cy)")
                            depth_scale = gr.Number(value=1000.0, label="Depth scale")

                        with gr.Accordion("🔧 Segmentation 3D", open=True):
                            gr.Markdown("**1. Suppression plans (table, murs)**")
                            plane_threshold = gr.Slider(0.001, 0.01, value=0.002, step=0.001, label="Seuil plan (m)")
                            max_planes = gr.Slider(1, 5, value=3, step=1, label="Nombre max de plans")
                            gr.Markdown("**2. Clustering 3D (DBSCAN)**")
                            dbscan_eps = gr.Slider(0.005, 0.05, value=0.015, step=0.001, label="Distance clustering (m)")
                            dbscan_min_points = gr.Slider(100, 1000, value=300, step=50, label="Points min par cluster")
                            gr.Markdown("**3. Sphere Fitting (RANSAC)**")
                            ransac_iterations = gr.Slider(500, 5000, value=2000, step=100, label="Itérations RANSAC")
                            ransac_threshold = gr.Slider(1.0, 10.0, value=3.0, step=0.5, label="Seuil RANSAC (mm)")

                        with gr.Accordion("🚀 Optimisations avancées", open=True):
                            use_optimization = gr.Checkbox(value=True,
                                label="Optimisation non-linéaire (Levenberg-Marquardt)",
                                info="Affine le fit après RANSAC (+0.2-0.4mm précision)")
                            use_outlier_removal = gr.Checkbox(value=True,
                                label="Filtrage statistique des outliers",
                                info="Supprime points aberrants (+0.3-0.5mm précision)")

                        process_btn = gr.Button("🚀 ANALYSER", variant="primary", size="lg")
                        process_status = gr.Markdown("")

                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Résultats")
                        summary_text = gr.Markdown()
                        output_image = gr.Image(label="Image annotée")
                        results_table = gr.Dataframe(label="Mesures détaillées", wrap=True)
                        export_btn = gr.Button("📥 Exporter en CSV", variant="secondary")
                        export_msg = gr.Markdown("")

            with gr.Tab("📊 Mon tableau de bord", visible=False) as dashboard_tab:
                with gr.Column():
                    gr.Markdown("### 📈 Mes statistiques")
                    stats_html = gr.HTML()
                    refresh_stats_btn = gr.Button("🔄 Actualiser", variant="secondary")

                    gr.Markdown("### 📋 Historique de mes analyses")
                    history_table = gr.Dataframe(
                        label="Dernières analyses",
                        wrap=True,
                        interactive=False
                    )
                    refresh_history_btn = gr.Button("🔄 Actualiser", variant="secondary")

        with gr.Accordion("💡 Conseils pour de meilleurs résultats", open=False):
            gr.Markdown("""
- **Distance caméra-oranges:** 40-50 cm
- **Espacement des oranges:** 2-3 cm minimum
- **Surface:** Table plane et stable
- **Éclairage:** Uniforme, sans ombres fortes
- **Qualité de la depth map:** Plus de détails = meilleure précision

**Précision attendue:**
- Excellente qualité (>90% confiance): ±1.0-1.5 mm
- Bonne qualité (75-90% confiance): ±1.5-2.0 mm
- Moyenne qualité (<75% confiance): ±2.0-3.0 mm
            """)

        gr.Markdown("---")
        with gr.Row():
            with gr.Column(scale=10):
                pass
            with gr.Column(scale=1, min_width=130):
                admin_btn = gr.Button("🔒 Admin", variant="secondary", size="sm")

    admin_authenticated = gr.State(False)

    with gr.Group(visible=False) as admin_login_group:
        gr.Markdown("## 🔐 Accès administrateur")
        with gr.Row():
            admin_pwd_input = gr.Textbox(label="Code d'accès", type="password", placeholder="Entrez le code admin", max_lines=1)
            admin_login_btn = gr.Button("Entrer", variant="primary")
        admin_login_msg = gr.Markdown("")

    with gr.Group(visible=False) as admin_dashboard:
        gr.Markdown("## 🛡️ Tableau de bord administrateur")
        with gr.Tabs():
            with gr.Tab("📊 Statistiques"):
                admin_stats_html = gr.HTML()
                admin_refresh_stats = gr.Button("🔄 Actualiser", variant="secondary")

            with gr.Tab("👥 Utilisateurs"):
                users_table = gr.Dataframe(label="Comptes enregistrés", wrap=True, interactive=False)
                admin_refresh_users = gr.Button("🔄 Actualiser", variant="secondary")

                gr.Markdown("### 🚫 Bannir / Débannir un utilisateur")
                with gr.Row():
                    ban_id_input = gr.Textbox(label="Nom d'utilisateur ou IP", placeholder="ex: jean_dupont", max_lines=1, scale=3)
                    ban_btn = gr.Button("🚫 Bannir", variant="stop", scale=1)
                    unban_btn = gr.Button("✅ Débannir", variant="secondary", scale=1)
                ban_result = gr.Markdown("")

            with gr.Tab("🖼️ Images uploadées"):
                uploads_table = gr.Dataframe(label="Journal des uploads", wrap=True, interactive=False)
                uploads_gallery = gr.Gallery(label="Aperçu", columns=4, height=400, object_fit="cover")
                admin_refresh_uploads = gr.Button("🔄 Actualiser", variant="secondary")

                gr.Markdown("### 🗑️ Supprimer des images")
                with gr.Row():
                    delete_filename_input = gr.Textbox(label="Nom du fichier", placeholder="ex: abc123_rgb.png", max_lines=1, scale=3)
                    delete_one_btn = gr.Button("🗑️ Supprimer", variant="stop", scale=1)
                with gr.Row():
                    delete_all_btn = gr.Button("🗑️ Supprimer TOUTES les images", variant="stop")
                delete_upload_msg = gr.Markdown("")

        admin_logout_btn = gr.Button("🔓 Se déconnecter de l'admin", variant="secondary", size="sm")

    def do_login(username, password, request: gr.Request):
        ip = ""
        try:
            ip = request.client.host if request and request.client else ""
        except:
            pass

        user, err = login_user(username, password, ip)
        if err:
            return None, gr.update(visible=True), gr.update(visible=False), f"<div class='error-message'>{err}</div>", ""

        token = create_session(user["id"], ip)
        greeting = f"<div style='padding:0.5rem;background:#f0fdf4;border:1px solid #86efac;border-radius:8px'>Connecté en tant que <strong>{user['username']}</strong></div>"

        return (token,
                gr.update(visible=False),
                gr.update(visible=True),
                "",
                greeting)

    login_btn.click(
        fn=do_login,
        inputs=[login_username, login_password],
        outputs=[session_token, auth_screen, main_app, login_msg, user_greeting]
    )

    def do_register(username, email, password, confirm, request: gr.Request):
        ip = ""
        try:
            ip = request.client.host if request and request.client else ""
        except:
            pass

        if password != confirm:
            return None, gr.update(visible=True), gr.update(visible=False), \
                   "<div class='error-message'>Les mots de passe ne correspondent pas.</div>", ""

        user, err = register_user(username, email, password, ip)
        if err:
            return None, gr.update(visible=True), gr.update(visible=False), f"<div class='error-message'>{err}</div>", ""

        if user.get("needs_verification"):
            msg = f"""<div class='success-message'>
                <strong>Compte créé avec succès!</strong><br>
                Un e-mail de vérification a été envoyé à <strong>{email}</strong>.<br>
                Veuillez vérifier votre boîte de réception (et vos spams) avant de vous connecter.
            </div>"""
            return None, gr.update(visible=True), gr.update(visible=False), msg, ""

        token = create_session(user["id"], ip)
        greeting = f"<div style='padding:0.5rem;background:#f0fdf4;border:1px solid #86efac;border-radius:8px'>Bienvenue <strong>{user['username']}</strong>!</div>"

        return (token,
                gr.update(visible=False),
                gr.update(visible=True),
                "",
                greeting)

    reg_btn.click(
        fn=do_register,
        inputs=[reg_username, reg_email, reg_password, reg_confirm],
        outputs=[session_token, auth_screen, main_app, reg_msg, user_greeting]
    )

    def do_password_reset_request(email):
        success, msg = request_password_reset(email)
        if success:
            return f"<div class='success-message'>{msg}</div>"
        return f"<div class='error-message'>{msg}</div>"

    reset_request_btn.click(
        fn=do_password_reset_request,
        inputs=[reset_email],
        outputs=[reset_request_msg]
    )

    def do_password_reset_confirm(token, new_password):
        success, msg = reset_password(token, new_password)
        if success:
            return f"<div class='success-message'>{msg}</div>"
        return f"<div class='error-message'>{msg}</div>"

    reset_confirm_btn.click(
        fn=do_password_reset_confirm,
        inputs=[reset_token, reset_new_password],
        outputs=[reset_confirm_msg]
    )

    def do_app_logout():
        return None, gr.update(visible=True), gr.update(visible=False), "", False, \
               gr.update(visible=False), gr.update(visible=False)

    app_logout_btn.click(
        fn=do_app_logout,
        inputs=[],
        outputs=[session_token, auth_screen, main_app, user_greeting,
                 admin_authenticated, admin_login_group, admin_dashboard]
    )

    def do_process(rgb_img, depth_img, fx, fy, cx, cy, depth_scale,
                   plane_threshold, max_planes, dbscan_eps, dbscan_min_points,
                   ransac_iterations, ransac_threshold,
                   use_optimization, use_outlier_removal, token, request: gr.Request):

        user = validate_session(token)
        if not user:
            return None, None, "<div class='error-message'>Session expirée. Veuillez vous reconnecter.</div>", "", None

        try:
            ip = request.client.host if request and request.client else ""
            if ip:
                update_user_ip(user["id"], ip)
        except:
            pass

        status_msg = "<div class='info-box'>⏳ Analyse en cours, veuillez patienter...</div>"

        annotated, df, summary = process_oranges(
            rgb_img, depth_img, fx, fy, cx, cy, depth_scale,
            plane_threshold, max_planes, dbscan_eps, dbscan_min_points,
            ransac_iterations, ransac_threshold,
            use_optimization, use_outlier_removal,
            user["id"]
        )

        if annotated is not None:
            status_msg = "<div class='success-message'>✅ Analyse terminée avec succès!</div>"
        else:
            status_msg = "<div class='error-message'>❌ L'analyse a échoué. Vérifiez vos images.</div>"

        return annotated, df, summary, status_msg, df

    process_btn.click(
        fn=do_process,
        inputs=[rgb_input, depth_input, fx, fy, cx, cy, depth_scale,
                plane_threshold, max_planes, dbscan_eps, dbscan_min_points,
                ransac_iterations, ransac_threshold,
                use_optimization, use_outlier_removal, session_token],
        outputs=[output_image, results_table, summary_text, process_status, gr.State()]
    )

    def export_to_csv(df, token):
        user = validate_session(token)
        if not user or df is None or df.empty:
            return "<div class='error-message'>Aucune donnée à exporter.</div>"

        try:
            export_path = UPLOADS_DIR / f"export_{user['username']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(export_path, index=False)
            return f"<div class='success-message'>✅ Export réussi: {export_path.name}</div>"
        except Exception as e:
            return f"<div class='error-message'>❌ Erreur lors de l'export: {e}</div>"

    def show_dashboard(token):
        user = validate_session(token)
        if not user:
            return gr.update(selected=0), "", pd.DataFrame()

        stats = get_user_stats(user["id"])
        stats_html_content = f"""
<div class="dashboard-grid">
  <div class="stat-card">
    <div style="font-size:2rem">🔬</div>
    <div style="font-size:1.8rem;font-weight:700;color:#667eea">{stats['total_analyses']}</div>
    <div style="color:#6b7280;font-size:0.9rem">Analyses</div>
  </div>
  <div class="stat-card">
    <div style="font-size:2rem">🍊</div>
    <div style="font-size:1.8rem;font-weight:700;color:#ea580c">{stats['total_oranges']}</div>
    <div style="color:#6b7280;font-size:0.9rem">Oranges</div>
  </div>
  <div class="stat-card">
    <div style="font-size:2rem">📏</div>
    <div style="font-size:1.8rem;font-weight:700;color:#9333ea">{stats['avg_diameter']} mm</div>
    <div style="color:#6b7280;font-size:0.9rem">Diamètre moyen</div>
  </div>
</div>
        """

        analyses = get_user_analyses(user["id"], limit=50)
        if analyses:
            history_df = pd.DataFrame([{
                "Date": a["timestamp"],
                "Oranges détectées": a["num_oranges"],
                "Diamètre moyen (mm)": round(a["avg_diameter"], 2) if a["avg_diameter"] else "—",
                "Confiance (%)": round(a["avg_confidence"], 1) if a["avg_confidence"] else "—"
            } for a in analyses])
        else:
            history_df = pd.DataFrame(columns=["Date", "Oranges détectées", "Diamètre moyen (mm)", "Confiance (%)"])

        return gr.update(selected=1), stats_html_content, history_df

    dashboard_btn.click(
        fn=show_dashboard,
        inputs=[session_token],
        outputs=[main_tabs, stats_html, history_table]
    )

    refresh_stats_btn.click(
        fn=show_dashboard,
        inputs=[session_token],
        outputs=[main_tabs, stats_html, history_table]
    )

    refresh_history_btn.click(
        fn=lambda token: show_dashboard(token)[2],
        inputs=[session_token],
        outputs=[history_table]
    )

    def toggle_admin_panel(current_auth):
        if current_auth:
            return gr.update(visible=False), gr.update(visible=True)
        return gr.update(visible=True), gr.update(visible=False)

    admin_btn.click(
        fn=toggle_admin_panel,
        inputs=[admin_authenticated],
        outputs=[admin_login_group, admin_dashboard]
    )

    def do_admin_login(pwd, current_auth):
        if verify_admin_password(pwd):
            s = admin_get_stats()
            return (True,
                    gr.update(visible=False), gr.update(visible=True), "",
                    admin_stats_html(s), admin_get_users(), admin_get_uploads_df(), admin_get_images_list())
        return (current_auth, gr.update(visible=True), gr.update(visible=False),
                "<div class='error-message'>Code incorrect.</div>",
                gr.update(), gr.update(), gr.update(), gr.update())

    admin_login_btn.click(
        fn=do_admin_login,
        inputs=[admin_pwd_input, admin_authenticated],
        outputs=[admin_authenticated, admin_login_group, admin_dashboard,
                 admin_login_msg, admin_stats_html, users_table, uploads_table, uploads_gallery]
    )

    def do_admin_logout():
        return False, gr.update(visible=False), gr.update(visible=False), ""

    admin_logout_btn.click(
        fn=do_admin_logout,
        inputs=[],
        outputs=[admin_authenticated, admin_login_group, admin_dashboard, admin_pwd_input]
    )

    admin_refresh_stats.click(fn=lambda: admin_stats_html(admin_get_stats()), outputs=[admin_stats_html])
    admin_refresh_users.click(fn=admin_get_users, outputs=[users_table])
    admin_refresh_uploads.click(
        fn=lambda: (admin_get_uploads_df(), admin_get_images_list()),
        outputs=[uploads_table, uploads_gallery]
    )

    def delete_one_fn(filename):
        msg, table, gallery = admin_delete_upload(filename)
        return f"<div class='success-message'>{msg}</div>", table, gallery, ""

    delete_one_btn.click(
        fn=delete_one_fn,
        inputs=[delete_filename_input],
        outputs=[delete_upload_msg, uploads_table, uploads_gallery, delete_filename_input]
    )

    def delete_all_fn():
        msg, table, gallery = admin_delete_all_uploads()
        return f"<div class='warning-message'>{msg}</div>", table, gallery

    delete_all_btn.click(
        fn=delete_all_fn,
        inputs=[],
        outputs=[delete_upload_msg, uploads_table, uploads_gallery]
    )

    def ban_fn(identifier):
        msg = admin_ban_user(identifier, ban=True)
        return f"<div class='success-message'>{msg}</div>", admin_get_users()

    def unban_fn(identifier):
        msg = admin_ban_user(identifier, ban=False)
        return f"<div class='success-message'>{msg}</div>", admin_get_users()

    ban_btn.click(fn=ban_fn, inputs=[ban_id_input], outputs=[ban_result, users_table])
    unban_btn.click(fn=unban_fn, inputs=[ban_id_input], outputs=[ban_result, users_table])

_dev_domain = os.environ.get("REPLIT_DEV_DOMAIN", "")
if not _dev_domain:
    _prod_domains = os.environ.get("REPLIT_DOMAINS", "")
    _root_path = f"https://{_prod_domains.split(',')[0].strip()}" if _prod_domains else ""
else:
    _root_path = ""

if __name__ == "__main__":
    logger.info("Starting Orange Measurement Application...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=5000,
        root_path=_root_path,
        allowed_paths=["uploads"],
    )
