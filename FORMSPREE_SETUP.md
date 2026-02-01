# Formspree Email Setup (EASIEST METHOD!)

## Why Formspree?

✅ **No SMTP configuration needed**
✅ **No app passwords required**
✅ **No firewall issues**
✅ **Works instantly**
✅ **Free tier: 50 submissions/month**

Perfect for personal use!

## Setup (3 Steps - Takes 2 Minutes!)

### Step 1: Get Your Formspree Endpoint

You already have one: `https://formspree.io/f/xlgnwblz`

Or create a new one at: https://formspree.io/

1. Sign up for free
2. Create a new form
3. Copy the endpoint URL (looks like: `https://formspree.io/f/xxxxxxxx`)

### Step 2: Update Your `.env` File

Open `.env` and set:

```bash
FORMSPREE_ENDPOINT=https://formspree.io/f/xlgnwblz
```

### Step 3: Run the Survey Bot

```bash
python run_web.py
```

That's it! 🎉

## How It Works

When you run a survey:
1. Bot extracts coupon code (e.g., `7091413`)
2. Sends data to Formspree
3. Formspree forwards email to your inbox
4. You receive:
   - Subject: "Your Survey Coupon Code: 7091413"
   - Body: Coupon code, store name, survey URL
   - Screenshot path

## Email Format

You'll receive emails like this:

```
From: Formspree <noreply@formspree.io>
To: your.email@gmail.com
Subject: Your Survey Coupon Code: 7091413

Survey Bot - Coupon Code Notification

Coupon Code: 7091413
Store: McDonald's
Survey URL: https://mcdvoice.com
Screenshot: screenshots/coupon_7091413_20260129.png

This code was automatically extracted from your survey.
Use it at your next visit!
```

## Pricing

**Free Tier**: 50 submissions/month
**Perfect for**: Personal survey automation

Need more? Upgrade at https://formspree.io/plans

## Priority

The bot checks email methods in this order:

1. **Formspree** (if `FORMSPREE_ENDPOINT` is set) ⭐ EASIEST
2. **SMTP** (if `SEND_EMAIL_VIA_SMTP=true`)
3. **Survey website** (default - McDonald's sends email themselves)

## Testing

1. Set your Formspree endpoint in `.env`
2. Run a survey with your email
3. Check your inbox!

## Troubleshooting

### "Formspree returned status 400"
- Check your endpoint URL is correct
- Make sure you copied the full URL including `/f/`

### "Formspree error: timeout"
- Check your internet connection
- Formspree might be down (rare) - try again

### Not receiving emails?
- Check your spam folder
- Verify your Formspree account email is correct
- Go to https://formspree.io/ and check your submissions

## Advantages Over SMTP

| Feature | Formspree | SMTP |
|---------|-----------|------|
| Setup time | 2 minutes | 15+ minutes |
| Configuration | 1 line | 5+ lines |
| App passwords | Not needed | Required |
| Firewall issues | None | Common |
| Maintenance | Zero | Sometimes breaks |
| Free tier | 50/month | Unlimited |

## Ready to Use!

Your `.env` is already configured with:
```bash
FORMSPREE_ENDPOINT=https://formspree.io/f/xlgnwblz
```

Just run the bot and it will automatically send coupon codes to your email! 🚀
