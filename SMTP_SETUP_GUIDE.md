# Email Setup Guide for Survey Bot

## Quick Setup

### For Gmail Users (Recommended)

1. **Enable 2-Factor Authentication** on your Google account
   - Go to: https://myaccount.google.com/security

2. **Create an App Password**
   - Go to: https://myaccount.google.com/apppasswords
   - Select "Mail" and "Windows Computer"
   - Google will generate a 16-character password
   - **Copy this password** (you'll only see it once)

3. **Update your `.env` file**:
   ```bash
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USER=your.email@gmail.com
   SMTP_PASSWORD=your-16-char-app-password
   SMTP_FROM_NAME=Survey Bot
   ```

### For Outlook/Hotmail Users

```bash
SMTP_HOST=smtp.office365.com
SMTP_PORT=587
SMTP_USER=your.email@outlook.com
SMTP_PASSWORD=your-password
SMTP_FROM_NAME=Survey Bot
```

### For Yahoo Mail Users

```bash
SMTP_HOST=smtp.mail.yahoo.com
SMTP_PORT=587
SMTP_USER=your.email@yahoo.com
SMTP_PASSWORD=your-app-password
SMTP_FROM_NAME=Survey Bot
```

**Note**: Yahoo also requires an app-specific password.
Generate one at: https://login.yahoo.com/account/security

## How It Works

When you run a survey:
1. Bot extracts the coupon code (e.g., `7091413`)
2. Takes a screenshot of the coupon page
3. Sends an email to the address you provided with:
   - Coupon code in the subject and body
   - Screenshot attached (if available)
   - Store information from the receipt

## Testing Email Configuration

Test if your email is configured correctly:

```bash
cd "J:\Project Files\MyPython\autoReceipt"
PYTHONPATH=src python -c "
import os
from dotenv import load_dotenv
load_dotenv()

smtp_host = os.environ.get('SMTP_HOST')
smtp_user = os.environ.get('SMTP_USER')
smtp_password = os.environ.get('SMTP_PASSWORD')

if smtp_host and smtp_user and smtp_password:
    print('[OK] SMTP configuration found!')
    print(f'Host: {smtp_host}')
    print(f'User: {smtp_user}')
    print(f'Password: {'*' * len(smtp_password)} (hidden)')
else:
    print('[ERROR] Missing SMTP configuration in .env file')
"
```

## Troubleshooting

### "SMTP configuration incomplete" Error
- Check that all required fields are filled in `.env`
- Make sure there are no quotes around values
- Restart the server after editing `.env`

### "Authentication failed" Error
- **Gmail**: Make sure you're using an App Password, NOT your regular password
- **Yahoo**: Make sure you're using an App Password
- **Outlook**: Regular password should work, but check if 2FA is enabled

### "Connection refused" Error
- Check your firewall settings
- Verify the SMTP host and port are correct
- Make sure you have internet connectivity

## Email Template Preview

The bot sends emails in this format:

```
Subject: Your Survey Coupon Code: 7091413

Hi!

Your survey has been completed successfully!

🎉 Your Coupon Code: 7091413

Store: McDonald's
Survey Date: 2026-01-29
Extraction Method: Element Search
Confidence: 95%

[Screenshot attached]

This code was automatically extracted from your survey completion.
Use it at your next visit!

---
Sent by Survey Bot
```

## Security Notes

- ✅ Never commit your `.env` file to version control
- ✅ Use app-specific passwords when possible
- ✅ Keep your SMTP credentials secure
- ✅ The bot uses TLS encryption for email sending
