"""
Formspree email sender - Simple email delivery using Formspree API.

This is an alternative to SMTP that doesn't require email server configuration.
Just use your Formspree form endpoint!
"""
import logging
import httpx
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class FormspreeSender:
    """Send emails via Formspree API."""

    def __init__(self, formspree_endpoint: str):
        """
        Initialize Formspree sender.

        Args:
            formspree_endpoint: Your Formspree form endpoint URL
                                (e.g., https://formspree.io/f/xlgnwblz)
        """
        self.endpoint = formspree_endpoint

    async def send_coupon_email(
        self,
        recipient_email: str,
        coupon_code: str,
        store_name: Optional[str] = None,
        survey_url: Optional[str] = None,
        screenshot_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send coupon code via Formspree.

        Args:
            recipient_email: Email address to send to
            coupon_code: The coupon/validation code
            store_name: Name of the store (e.g., McDonald's)
            survey_url: Original survey URL
            screenshot_path: Path to screenshot (optional)

        Returns:
            Dict with 'success' and 'message' keys
        """
        logger.info(f"Sending coupon via Formspree to {recipient_email}")

        # Prepare the message
        message_lines = [
            f"Survey Bot - Coupon Code Notification",
            f"",
            f"Coupon Code: {coupon_code}",
        ]

        if store_name:
            message_lines.append(f"Store: {store_name}")
        if survey_url:
            message_lines.append(f"Survey URL: {survey_url}")
        if screenshot_path:
            message_lines.append(f"Screenshot: {screenshot_path}")

        message_lines.append("")
        message_lines.append("This code was automatically extracted from your survey.")
        message_lines.append("Use it at your next visit!")

        # Prepare form data
        form_data = {
            "email": recipient_email,
            "subject": f"Your Survey Coupon Code: {coupon_code}",
            "message": "\n".join(message_lines),
            "coupon_code": coupon_code,
        }

        if store_name:
            form_data["store"] = store_name

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.endpoint,
                    data=form_data,
                    headers={
                        "Accept": "application/json"
                    },
                    timeout=30.0
                )

                if response.status_code == 200:
                    logger.info(f"Successfully sent coupon to {recipient_email} via Formspree")
                    return {
                        "success": True,
                        "message": "Email sent successfully via Formspree"
                    }
                else:
                    error_msg = f"Formspree returned status {response.status_code}"
                    logger.error(error_msg)
                    return {
                        "success": False,
                        "message": error_msg
                    }

        except Exception as e:
            error_msg = f"Formspree error: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg
            }
