Orange Diameter Measurement App
Overview
A Gradio web application for measuring orange diameters using 3D depth maps, with a full admin panel for user management and analytics.

Pipeline
Depth Map → 3D Point Cloud → Multi-Plane RANSAC → DBSCAN Clustering → Statistical Outlier Removal → RANSAC Sphere Fitting → Levenberg-Marquardt Optimization

Expected precision: ±1.0-1.5 mm

Project Structure
app.py — Main application (Gradio UI + 3D processing + admin panel)
requirements.txt — Python dependencies
admin.db — SQLite database (auto-created, tracks users/uploads/analyses)
uploads/ — Saved user images (auto-created)
Tech Stack
Python 3.12
Gradio 5.23.0 — Web UI
Open3D — 3D point cloud processing
OpenCV (headless) — Image processing and annotation
NumPy / SciPy — Numerical computation
Pandas — Results table
SQLite — Admin database (built-in)
Admin Panel
Accessible via the 🔒 Admin button at the bottom of the page.

Password: ORANGEADMIN
Features:
📊 Statistics: total users, oranges analyzed, average diameter, analyses today
👥 User management: view all users with IP, email, first/last seen
🚫 Ban/unban users by IP address
🖼️ Image gallery: view all uploaded images with metadata
User Tracking
IP address captured automatically on every analysis
Optional email field at the top of the main UI
Uploaded RGB images saved to uploads/ directory
All analyses logged (orange count, avg diameter, confidence)
Banned users are blocked from running analyses
Running the App
python app.py

Serves on 0.0.0.0:5000.

System Dependencies (Nix)
Required for Open3D on Nix/production:

xorg.libX11, libGL, libgcc, gcc-unwrapped
llvmPackages.libcxx, libcap
The startup routine in app.py also symlinks libudev.so.1 and libcap.so.2 from the system path automatically.

Deployment
Configured for autoscale deployment running python app.py.
