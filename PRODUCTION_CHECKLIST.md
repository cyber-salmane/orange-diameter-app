# Production Deployment Checklist

Use this checklist before deploying to production.

## Security

### Authentication & Authorization
- [ ] Admin password set via environment variable (not hardcoded)
- [ ] Strong admin password (16+ characters, mixed case, numbers, symbols)
- [ ] Email verification enabled in config
- [ ] Session timeout appropriate for use case (default: 24 hours)
- [ ] Rate limiting configured (default: 5 attempts per 15 minutes)
- [ ] Reviewed and tested password reset flow

### Data Protection
- [ ] HTTPS enabled (SSL certificate configured)
- [ ] Database connection encrypted (if using remote database)
- [ ] File upload restrictions tested (size, type)
- [ ] Path traversal protection verified
- [ ] SQL injection prevention confirmed (parameterized queries)
- [ ] No sensitive data in logs

### Session Management
- [ ] Sessions use secure tokens
- [ ] Session expiration works correctly
- [ ] Old sessions cleaned up automatically
- [ ] Session tokens not exposed in URLs

## Infrastructure

### Database
- [ ] Migrated from SQLite to PostgreSQL/MySQL (recommended)
- [ ] Database backups configured
- [ ] Backup restoration tested
- [ ] Database credentials secured (environment variables)
- [ ] Database indexes created
- [ ] Database connection pooling configured

### Email Service
- [ ] SMTP server configured (or SendGrid/AWS SES)
- [ ] Email templates reviewed
- [ ] Verification emails tested
- [ ] Password reset emails tested
- [ ] Email rate limiting configured
- [ ] SPF/DKIM records configured

### File Storage
- [ ] Uploads directory writable
- [ ] Disk space monitoring enabled
- [ ] File cleanup job scheduled (old files)
- [ ] Consider S3/cloud storage for scalability
- [ ] File permissions secured (not world-readable)

### Web Server
- [ ] Reverse proxy configured (nginx/Apache)
- [ ] Rate limiting at proxy level
- [ ] Request size limits configured
- [ ] Static file caching enabled
- [ ] Compression enabled (gzip)

## Monitoring & Logging

### Logging
- [ ] Log level appropriate (INFO in production)
- [ ] Logs rotated (prevent disk fill)
- [ ] Logs aggregated (ELK/CloudWatch/etc)
- [ ] No passwords or tokens in logs
- [ ] Error tracking configured (Sentry/etc)

### Monitoring
- [ ] Uptime monitoring enabled
- [ ] Performance metrics tracked
- [ ] Disk space alerts configured
- [ ] CPU/memory alerts configured
- [ ] Database performance monitored
- [ ] Failed login attempts monitored

### Alerts
- [ ] Alert on repeated failed logins
- [ ] Alert on database errors
- [ ] Alert on disk space issues
- [ ] Alert on application errors
- [ ] Alert on unusual traffic patterns

## Performance

### Optimization
- [ ] Database queries optimized
- [ ] Database indexes created
- [ ] Gradio components optimized
- [ ] Image processing tested under load
- [ ] Session cleanup scheduled
- [ ] File cleanup scheduled

### Scalability
- [ ] Load testing completed
- [ ] Concurrent user limit known
- [ ] Database connection pool sized
- [ ] Auto-scaling configured (if cloud)
- [ ] CDN configured for static assets (optional)

## Testing

### Functional Testing
- [ ] Registration flow tested
- [ ] Email verification tested
- [ ] Login flow tested
- [ ] Password reset tested
- [ ] Session expiration tested
- [ ] Rate limiting tested
- [ ] File upload tested (valid/invalid files)
- [ ] Orange analysis tested
- [ ] Admin panel tested
- [ ] User banning tested

### Security Testing
- [ ] Attempted SQL injection (should fail)
- [ ] Attempted path traversal (should fail)
- [ ] Attempted session hijacking (should fail)
- [ ] Tested with invalid file types (should reject)
- [ ] Tested with oversized files (should reject)
- [ ] Attempted brute force login (should rate limit)

### Performance Testing
- [ ] Tested with multiple concurrent users
- [ ] Tested with large depth maps
- [ ] Tested with many oranges in one image
- [ ] Database query performance acceptable
- [ ] Page load times acceptable

## Documentation

### User Documentation
- [ ] README updated with production info
- [ ] User guide created or updated
- [ ] FAQ created
- [ ] Contact information provided

### Technical Documentation
- [ ] Architecture documented
- [ ] API endpoints documented (if any)
- [ ] Database schema documented
- [ ] Configuration options documented
- [ ] Deployment process documented

### Operations Documentation
- [ ] Backup/restore procedure documented
- [ ] Monitoring setup documented
- [ ] Troubleshooting guide created
- [ ] Incident response plan created
- [ ] Rollback procedure documented

## Compliance & Legal

### Data Privacy
- [ ] GDPR compliance reviewed (if applicable)
- [ ] Privacy policy created
- [ ] Terms of service created
- [ ] Data retention policy defined
- [ ] User data export capability (if required)
- [ ] User data deletion capability (if required)

### Security Compliance
- [ ] Security audit completed
- [ ] Penetration testing completed
- [ ] Vulnerability scan completed
- [ ] Dependencies updated (no known vulnerabilities)

## Deployment

### Pre-Deployment
- [ ] Code reviewed
- [ ] All tests passing
- [ ] Staging environment tested
- [ ] Database migration tested
- [ ] Rollback plan prepared
- [ ] Downtime communicated (if any)

### Configuration
- [ ] Environment variables set
- [ ] Config files reviewed
- [ ] Secret management configured
- [ ] API keys secured
- [ ] Feature flags configured (if applicable)

### Post-Deployment
- [ ] Health check passing
- [ ] Key user flows tested
- [ ] Logs reviewed for errors
- [ ] Performance metrics normal
- [ ] Backup verified
- [ ] Monitoring alerts active

## Maintenance

### Regular Tasks
- [ ] Weekly log review scheduled
- [ ] Monthly security update check scheduled
- [ ] Quarterly dependency update scheduled
- [ ] Database backup verification scheduled
- [ ] Old file cleanup automated

### Emergency Procedures
- [ ] Emergency contact list created
- [ ] Escalation procedure defined
- [ ] On-call rotation scheduled (if applicable)
- [ ] Incident response playbook created

## Environment-Specific

### Development
- [ ] Email simulation documented
- [ ] Test users created
- [ ] Debug logging enabled
- [ ] Realistic test data available

### Staging
- [ ] Mirrors production configuration
- [ ] Real email service (test mode)
- [ ] Production-like data volume
- [ ] Load testing possible

### Production
- [ ] All checklist items above completed
- [ ] Monitoring active
- [ ] Backups configured
- [ ] Support contact available

## Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Developer | | | |
| Security | | | |
| Operations | | | |
| Product Owner | | | |

---

## Quick Command Reference

### Set Admin Password (Production)
```bash
export ADMIN_PASSWORD="$(openssl rand -base64 32)"
echo $ADMIN_PASSWORD > admin_password.txt  # Save securely!
```

### Check Logs
```bash
tail -f app.log | grep -E "ERROR|WARNING"
```

### Database Backup
```bash
sqlite3 admin.db .dump > backup_$(date +%Y%m%d).sql
```

### Check Disk Space
```bash
du -sh uploads/
df -h
```

### Test Email Configuration
```bash
python3 -c "from utils import send_verification_email; send_verification_email('test@example.com', 'test123')"
```

---

**Last Updated:** 2026-04-04
**Checklist Version:** 2.0
**Application Version:** 2.0.0
