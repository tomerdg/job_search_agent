"""Daily digest via Telegram and/or email. Falls back to printing."""
from __future__ import annotations

import html
import json
import logging
import os
import smtplib
from email.mime.text import MIMEText

import requests

log = logging.getLogger(__name__)


def format_digest(rows, total_new: int) -> str:
    if not rows:
        return f"🔎 סוכן משרות: {total_new} משרות חדשות היום, אין התאמה חזקה."
    lines = [f"🔎 <b>3 המשרות המתאימות היום</b> (מתוך {total_new} חדשות)\n"]
    for i, r in enumerate(rows, 1):
        a = json.loads(r["analysis"] or "{}")
        lines.append(
            f"<b>{i}. {html.escape(r['title'])}</b> · {html.escape(r['company'])}\n"
            f"📍 {html.escape(r['location'] or 'לא צוין')} · ציון {r['score']}/100\n"
            f"{html.escape(a.get('summary', ''))}\n"
            f"✅ {html.escape('; '.join(a.get('strengths', [])))}\n"
            f"⚠️ {html.escape('; '.join(a.get('gaps', [])))}\n"
            f"{html.escape(r['url'] or '')}\n"
        )
    return "\n".join(lines)


def send_telegram(text: str) -> bool:
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat):
        return False
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", timeout=30, json={
        "chat_id": chat, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True,
    })
    r.raise_for_status()
    return True


def send_email(text: str) -> bool:
    host, to = os.environ.get("SMTP_HOST"), os.environ.get("EMAIL_TO")
    if not (host and to):
        return False
    user, pwd = os.environ.get("SMTP_USER", ""), os.environ.get("SMTP_PASSWORD", "")
    msg = MIMEText(text.replace("\n", "<br>"), "html", "utf-8")
    msg["Subject"] = "סוכן משרות: ההתאמות של היום"
    msg["From"] = os.environ.get("EMAIL_FROM", user)
    msg["To"] = to
    with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", "587")), timeout=30) as s:
        s.starttls()
        if user:
            s.login(user, pwd)
        s.send_message(msg)
    return True


def notify(text: str) -> bool:
    """Send the digest. Never prints it: CI logs may be public."""
    sent = False
    for fn in (send_telegram, send_email):
        try:
            sent = fn(text) or sent
        except Exception as exc:  # noqa: BLE001
            log.error("%s failed: %s", fn.__name__, exc)
    if not sent:
        log.warning("No Telegram or email configured; digest not sent (use --no-notify to print it locally)")
    return sent
