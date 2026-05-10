# Security & Email Integration Guide

## Security Improvements

### Admin Password
- `ADMIN_PASSWORD` is now **required** and no longer has a built-in default.
- Generate a secure password and set it in `.env`:
  ```bash
  export ADMIN_PASSWORD="$(openssl rand -base64 32)"
  ```
- If not set, admin login will fail (returns `False`).

### Rate Limiting

Three levels of rate limiting are now enforced:

1. **Login attempts**: 5 failed attempts per IP/username within 15 minutes
2. **Registration attempts**: 5 per email/IP within 15 minutes
3. **Password reset requests**: 3 per email/IP within 60 minutes

These are tracked in the database with `registration_attempts` and `password_reset_requests` tables.

### Session Token Hashing

Session tokens are now stored as **hashed values** in the database:
- Each session has `token` (plaintext, sent to client) and `token_hash` (SHA256, stored in DB).
- During validation, tokens are hashed before checking against `token_hash`.
- Protects sessions if the database is compromised.

### Image Upload Validation

File validation now includes:
- **MIME type checking**: Validates `image/png` and `image/jpeg` only.
- **File signature verification**: Opens and verifies images with PIL before accepting uploads.
- Rejects invalid or suspicious files.

## Email Reliability

### SendGrid Integration (Recommended)

SendGrid is ideal for production transactional email:

1. Create a free SendGrid account: https://sendgrid.com
2. Get your API key from the dashboard
3. Set in `.env`:
   ```
   SENDGRID_API_KEY=SG.your_api_key_here
   SENDGRID_FROM_EMAIL=noreply@your-domain.com
   ```
4. The app will automatically use SendGrid when `SENDGRID_API_KEY` is set.

### SMTP Fallback

If SendGrid is not configured, the app falls back to SMTP:
- Gmail SMTP: `smtp.gmail.com:587`
- Outlook: `smtp-mail.outlook.com:587`
- Custom SMTP server

Set in `.env`:
```
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your.email@gmail.com
SMTP_PASSWORD=your_app_password
SMTP_FROM_EMAIL=your.email@gmail.com
```

### Retry Logic

Email sending now includes automatic retries:
- **3 retry attempts** with exponential backoff (1s, 2s, 4s)
- **Logs all failures** to help debug configuration issues
- Falls back to console simulation if no email provider is configured

### Email Simulation (Development)

If no email provider is configured, emails are printed to console:
```
[EMAIL SIMULATION] To: user@example.com
[EMAIL SIMULATION] Subject: Verify your email
```

This allows testing without setting up email credentials.

## Database Schema Updates

New tables for rate limiting:

```sql
CREATE TABLE registration_attempts (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL,
    ip TEXT NOT NULL,
    success INTEGER NOT NULL,
    timestamp TEXT NOT NULL
);

CREATE TABLE password_reset_requests (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL,
    ip TEXT NOT NULL,
    success INTEGER NOT NULL,
    timestamp TEXT NOT NULL
);
```

The `sessions` table now has:
- `token`: plaintext token (sent to client)
- `token_hash`: SHA256 hash (stored in DB)
- Both fields are indexed and unique

## Hugging Face Spaces Deployment

1. **Set ADMIN_PASSWORD in Spaces secrets:**
   ```
   ADMIN_PASSWORD=your-secure-password-here
   ```

2. **For SendGrid email**:
   ```
   SENDGRID_API_KEY=SG.your_key
   SENDGRID_FROM_EMAIL=noreply@your-domain.com
   ```

3. **Or for SMTP**:
   ```
   SMTP_USERNAME=your.email@gmail.com
   SMTP_PASSWORD=your_app_password
   ```

## Testing Security

Test rate limiting:
```bash
curl -X POST http://localhost:7860/register \
  -d '{"username":"test","email":"test@example.com","password":"BadPass123"}' && \
  # Repeat 5 times to trigger rate limit
```

Test image upload validation (via web UI):
- Upload a text file with `.png` extension → rejected
- Upload a valid PNG → accepted

Test session hashing:
- Login and note the session token
- Database `sessions.token_hash` should be a 64-character hex string
- Database `sessions.token` may be empty (for security)
