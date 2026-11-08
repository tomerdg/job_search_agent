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
            f"🆔 {r['job_id'][:8]}\n"
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


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def send_email(text: str) -> bool:
    host, to = _env("SMTP_HOST"), _env("EMAIL_TO")
    if not (host and to):
        return False
    if ":" in host:  # allow "smtp.gmail.com:587"
        host, port_s = host.rsplit(":", 1)
    else:
        port_s = _env("SMTP_PORT", "587")
    user = _env("SMTP_USER")
    pwd = _env("SMTP_PASSWORD").replace(" ", "")  # Gmail app passwords are shown with spaces
    msg = MIMEText(text.replace("\n", "<br>"), "html", "utf-8")
    msg["Subject"] = "סוכן משרות: ההתאמות של היום"
    msg["From"] = _env("EMAIL_FROM", user)
    msg["To"] = to

    stage = {"v": ""}

    def via_starttls(port: int):
        stage["v"] = "connect"
        with smtplib.SMTP(host, port, timeout=30) as s:
            stage["v"] = "ehlo"
            s.ehlo()
            stage["v"] = "starttls"
            s.starttls()
            s.ehlo()
            if user:
                stage["v"] = "login"
                s.login(user, pwd)
            stage["v"] = "send"
            s.send_message(msg)

    def via_ssl(port: int):
        stage["v"] = "connect"
        with smtplib.SMTP_SSL(host, port, timeout=30) as s:
            if user:
                stage["v"] = "login"
                s.login(user, pwd)
            stage["v"] = "send"
            s.send_message(msg)

    # Safe diagnostics (no secret values): helps spot a wrong host or password format.
    log.info("SMTP check: host_is_gmail=%s host_has_at=%s user_is_email=%s password_len=%d to_is_email=%s",
             host.lower() == "smtp.gmail.com", "@" in host, "@" in user, len(pwd), "@" in to)
    port = int(port_s)
    attempts = [(via_ssl, port)] if port == 465 else [(via_starttls, port), (via_ssl, 465)]
    last = None
    for fn, p in attempts:
        try:
            fn(p)
            log.info("Digest emailed via %s:%s", host, p)
            return True
        except Exception as exc:  # noqa: BLE001
            last = exc
            log.warning("SMTP %s:%s (%s) failed at %s: %s: %s", host, p, fn.__name__, stage["v"], type(exc).__name__, exc)
    raise RuntimeError(f"all SMTP attempts failed, last error: {last}")


def notify(text: str) -> bool:
    """Send the digest. Never prints it: CI logs may be public."""
    sent = False
    for fn in (send_telegram, send_email):
        try:
            sent = fn(text) or sent
        except Exception as exc:  # noqa: BLE001
            log.error("%s failed: %s", fn.__name__, exc)
    if not sent:
        log.warning("Digest not sent (no working Telegram/email). It will be retried on the next run.")
    return sent
