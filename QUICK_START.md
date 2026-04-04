# Quick Start Guide

## Installation (2 minutes)

```bash
# 1. Install dependencies
pip install bcrypt --user

# 2. Run the app
python3 app.py
```

Open: `http://localhost:5000`

## First User Registration

1. Click **"Créer un compte"** tab
2. Choose username (3-30 characters)
3. Enter email address
4. Create password (8+ chars, uppercase, lowercase, number)
5. **Check console** for verification link
6. Copy the token and verify your email

## Email Verification (Dev Mode)

Look for this in the console:
```
============================================================
VERIFICATION EMAIL SIMULATION
============================================================
To: your-email@example.com
...
Verification link: https://app.example.com/verify?token=ABC123...
============================================================
```

**For now:** Email verification is ENABLED by default.

**To disable:** Edit `config.py`:
```python
EMAIL_VERIFICATION_ENABLED = False
```

## Analyzing Oranges

1. Login
2. Go to "🔬 Analyser" tab
3. Upload RGB image
4. Upload depth map
5. Click "🚀 ANALYSER"
6. View results

## Viewing Dashboard

1. Click "📊 Mon tableau de bord" button (top right)
2. See your statistics
3. View analysis history
4. Export to CSV

## Admin Access

1. Click "🔒 Admin" button (bottom right)
2. Enter password (default: `ORANGEADMIN`)
3. View statistics, users, and uploads

## Common Tasks

### Reset a Password

1. Click "Mot de passe oublié ?"
2. Enter email
3. Check console for reset token
4. Enter token + new password

### Change Admin Password

```bash
export ADMIN_PASSWORD="your-new-password"
python3 app.py
```

### Ban a User (Admin)

1. Admin panel → Users tab
2. Enter username or IP
3. Click "🚫 Bannir"

### Delete Uploaded Images (Admin)

1. Admin panel → Images tab
2. Enter filename
3. Click "🗑️ Supprimer"

## Configuration Quick Reference

Edit `config.py`:

```python
ADMIN_PASSWORD = "ORANGEADMIN"          # Change this!
MAX_FILE_SIZE_MB = 10                   # Max upload size
SESSION_TIMEOUT_HOURS = 24              # How long users stay logged in
EMAIL_VERIFICATION_ENABLED = True       # Require email verification
MAX_LOGIN_ATTEMPTS = 5                  # Before rate limiting
CLEANUP_OLD_FILES_DAYS = 30            # File retention period
```

## Troubleshooting

### "Session expired"
→ Login again (sessions last 24 hours)

### "Email not verified"
→ Check console for verification link

### "Rate limited"
→ Wait 15 minutes (too many failed logins)

### "File too large"
→ Max 10MB (configurable in config.py)

### Import errors
→ Make sure all .py files are in the same directory

### "bcrypt not found"
→ `pip install bcrypt --user`

## Security Checklist

For production deployment:

- [ ] Change `ADMIN_PASSWORD` via environment variable
- [ ] Set up real email service (edit utils.py)
- [ ] Enable HTTPS
- [ ] Use PostgreSQL instead of SQLite
- [ ] Set up database backups
- [ ] Review rate limiting settings
- [ ] Enable monitoring/logging

## File Structure

```
.
├── app.py              # Main UI (run this)
├── config.py           # Settings
├── db.py               # Database
├── auth.py             # Login/register
├── processing.py       # Orange analysis
├── admin.py            # Admin panel
├── utils.py            # Helpers
├── requirements.txt    # Dependencies
├── admin.db            # Database (auto-created)
└── uploads/            # User files (auto-created)
```

## Need Help?

- **README.md** - Full user guide
- **UPGRADE_GUIDE.md** - Technical documentation
- **CHANGES.md** - What's new in v2.0

## Default Credentials

**No default users** - You create the first account

**Admin password:** `ORANGEADMIN` (change this!)

---

**Quick tip:** All emails are printed to console in development mode. Check terminal output for verification links and password reset tokens.
