# Orange Measurement App - Production Upgrade Guide

## Overview

Your Gradio-based orange measurement application has been completely refactored into a production-grade system with enterprise-level security, modular architecture, and enhanced user experience.

## What's New

### 1. Modular Architecture

The monolithic `app.py` has been split into focused modules:

- **config.py** - Application configuration and constants
- **db.py** - Database operations and queries
- **auth.py** - Authentication and authorization logic
- **processing.py** - 3D image processing and analysis
- **admin.py** - Admin panel functionality
- **utils.py** - Utility functions and helpers
- **app.py** - Main Gradio UI and event handlers

### 2. Security Enhancements

#### Authentication
- **bcrypt password hashing** replaces SHA256 (much more secure)
- **Email verification** on registration (simulated emails to console)
- **Password reset functionality** with token-based system
- **Session management** with automatic expiration (24 hours default)
- **Rate limiting** on login attempts (5 attempts per 15 minutes)
- **Password requirements**: minimum 8 characters, uppercase, lowercase, and number

#### Input Validation
- Email format validation
- Username validation (3-30 chars, alphanumeric + `-_`)
- File type restrictions (PNG, JPG, JPEG only)
- File size limits (10MB max)
- Sanitized filenames to prevent path traversal attacks

#### Database Security
- SQL injection prevention via parameterized queries
- Database indexes for performance
- Session token-based authentication
- Banned user checks

### 3. User Features

#### User Dashboard
- Personal statistics (total analyses, oranges analyzed, average diameter)
- Analysis history with timestamps
- CSV export functionality
- Clean, card-based UI design

#### Improved UX
- Modern gradient header
- Color-coded success/error/warning messages
- Loading states during processing
- Responsive design
- Better visual hierarchy
- Hover effects and transitions

### 4. Admin Panel Improvements

- Enhanced statistics display with verified users count
- Better table formatting
- Confirmation-style messages
- More secure access (hidden button)
- Search functionality ready

### 5. File Management

- Automatic cleanup of old files (30 days default)
- Consistent file naming with UUIDs
- File size tracking in database
- Upload history with metadata

### 6. Logging & Monitoring

- Structured logging throughout the application
- Action logging (registration, login, uploads, analysis)
- Error tracking
- Security event logging (failed logins, bans)

## Installation

### Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

**New dependency:** `bcrypt` for secure password hashing

### Configuration

Edit `config.py` to customize:

- `ADMIN_PASSWORD` - Admin panel password (use environment variable in production)
- `MAX_FILE_SIZE_MB` - Maximum upload size (default: 10MB)
- `SESSION_TIMEOUT_HOURS` - Session expiration (default: 24 hours)
- `CLEANUP_OLD_FILES_DAYS` - File retention period (default: 30 days)
- `EMAIL_VERIFICATION_ENABLED` - Toggle email verification (default: True)

### Environment Variables

Recommended production setup:

```bash
export ADMIN_PASSWORD="your-secure-password-here"
```

## Running the Application

```bash
python3 app.py
```

The app will:
1. Initialize the database (with new tables for sessions, password resets, etc.)
2. Clean up expired sessions
3. Clean up old login attempts
4. Clean up old uploaded files
5. Start the Gradio server on port 5000

## Database Schema Changes

The database now includes:

- **users** table: added `is_verified`, `verification_token`, `created_at`
- **password_resets** table: NEW - for password reset tokens
- **login_attempts** table: NEW - for rate limiting
- **sessions** table: NEW - for session management
- **uploads** table: added `file_size` column
- Multiple indexes for performance optimization

The old database will be automatically migrated on first run.

## User Workflow

### Registration
1. User fills out registration form
2. Password is validated (8+ chars, mixed case, number)
3. Email verification email is sent (to console in dev mode)
4. User must verify email before logging in

### Login
1. Username or email + password
2. Rate limiting check (max 5 failed attempts per 15 minutes)
3. Email verification check
4. Session token created (expires in 24 hours)
5. User redirected to main app

### Password Reset
1. User requests reset via email
2. Reset token generated (expires in 1 hour)
3. Reset link sent to email (to console in dev mode)
4. User submits token + new password
5. Password updated

## Email Simulation

Since this is a development/demo environment, emails are printed to the console:

```
============================================================
VERIFICATION EMAIL SIMULATION
============================================================
To: user@example.com
Subject: Vérifiez votre adresse e-mail

Cliquez sur ce lien pour vérifier votre compte:
https://app.example.com/verify?token=abc123...
============================================================
```

In production, replace `send_verification_email()` and `send_password_reset_email()` in `utils.py` with real SMTP integration.

## Security Best Practices

### In Production

1. **Use environment variables** for sensitive config:
   ```python
   ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "fallback")
   ```

2. **Enable HTTPS** (Gradio supports SSL certificates)

3. **Use a production database** (PostgreSQL, MySQL) instead of SQLite

4. **Set up real email service** (SendGrid, AWS SES, etc.)

5. **Monitor logs** for security events

6. **Regular backups** of the database

7. **Update dependencies** regularly for security patches

## Performance Optimizations

- Database indexes on frequently queried columns
- Session cleanup on startup
- Old file cleanup
- Efficient SQL queries with proper joins
- Minimal state in Gradio components

## API Changes

If you had any external integrations, note these changes:

- Sessions now use token-based authentication instead of user dict
- User IDs are UUIDs (strings) instead of auto-increment integers
- All passwords are bcrypt-hashed (can't be recovered, only reset)

## Troubleshooting

### "Session expired" errors
- Sessions expire after 24 hours (configurable in `config.py`)
- User needs to log in again

### Email not received
- Check console output for simulated emails
- In production, check SMTP logs

### Rate limiting
- Wait 15 minutes after 5 failed login attempts
- Check `login_attempts` table to clear if needed

### Import errors
- Ensure all modules are in the same directory
- Check Python path

## Backward Compatibility

The new app.py maintains full backward compatibility with existing features:

- All 3D processing functionality unchanged
- Admin panel features preserved
- Upload tracking maintained
- Analysis history available

Existing data in `admin.db` will be preserved. New columns will be added automatically.

## Migration from Old Version

1. **Backup your database**: `cp admin.db admin.db.backup`
2. **Backup uploads folder**: `cp -r uploads uploads_backup`
3. Replace old files with new modules
4. Install bcrypt: `pip install bcrypt`
5. Run the new app: `python3 app.py`

The database will auto-migrate on first run.

## Testing Checklist

- [ ] Registration with email verification
- [ ] Login with username
- [ ] Login with email
- [ ] Password reset flow
- [ ] Session expiration
- [ ] Rate limiting (5+ failed logins)
- [ ] File upload validation
- [ ] Orange analysis processing
- [ ] User dashboard
- [ ] CSV export
- [ ] Admin login
- [ ] User banning
- [ ] File deletion
- [ ] Statistics display

## Future Enhancements

Potential improvements for v2.0:

- OAuth2 integration (Google, GitHub login)
- Two-factor authentication (TOTP)
- Batch processing for multiple images
- REST API endpoints
- WebSocket for real-time progress
- Advanced analytics dashboard
- User roles and permissions
- API rate limiting
- Audit log export
- Data retention policies

## Support

For issues or questions:
1. Check application logs (console output)
2. Review error messages (now color-coded)
3. Check database integrity
4. Verify all dependencies installed

## License

Same as original project.

---

**Version:** 2.0.0
**Last Updated:** 2026-04-04
**Minimum Python:** 3.8+
