"""Format subject + body the way the classifier expects (structure helps SBERT)."""


def format_for_classifier(subject: str, body: str) -> str:
    subject = (subject or "").strip()
    body = (body or "").strip()
    if not subject and not body:
        return ""
    if not subject:
        return f"Body: {body}"
    if not body:
        return f"Subject: {subject}"
    return f"Subject: {subject}\nBody: {body}"
