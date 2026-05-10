# Hugging Face Spaces Deployment

This project can be deployed easily on Hugging Face Spaces using the Gradio runtime.

## Recommended preparation

1. Create a new public or private Space on Hugging Face.
2. Select `Gradio` and `Python` as the runtime.
3. Push this repository to the Space.

## Required environment variables

Set the following secret values in your Space settings:

- `APP_BASE_URL=https://<your-space-name>.hf.space`
- `SMTP_SERVER` (e.g. `smtp.gmail.com`)
- `SMTP_PORT` (e.g. `587`)
- `SMTP_USERNAME` (your email address or SMTP user)
- `SMTP_PASSWORD` (app password or SMTP password)
- `SMTP_FROM_EMAIL` (sender address shown in emails)
- `ADMIN_PASSWORD` (optional but recommended)

## Notes for email verification

- `APP_BASE_URL` must be the public Hugging Face Space URL.
- Verification links are generated from `APP_BASE_URL`.
- If this value is wrong, verification emails will not work.

## Local development

- Copy `.env.example` to `.env` and update values.
- Run locally with:

```bash
python3 app.py
```

- The app listens on `0.0.0.0:7860` by default.
- Run unit tests with:

```bash
pytest
```

- Build and run with Docker:

```bash
docker build -t orange-diameter-app .
docker run --rm -p 7860:7860 orange-diameter-app
```

## App behavior on Spaces

- The app now launches with `share=False`.
- Verification is handled by an authenticated `/verify` route.
- Password reset emails point to `/reset-password` and display a token.

## Recommended upgrades after deployment

- Use a managed database instead of SQLite for production.
- Store uploads in cloud object storage instead of local disk.
- Use a transactional email provider such as SendGrid, Mailgun, or AWS SES.
- Add automated tests and a CI workflow.
