import html
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.pipeline import EmailAnalyzer
from src.email_format import format_for_classifier

try:
    import dateparser
except ImportError:
    dateparser = None

# ---------- PAGE ----------
st.set_page_config(page_title="Email Overload Organizer", layout="wide")

analyzer = EmailAnalyzer()

# ---------- HELPERS ----------


def _confidence_display(result):
    adj = result.get("adjusted_confidence")
    raw = result.get("model_confidence")
    src = result.get("classifier_source") or ""
    if adj is not None:
        return f"{adj:.0%}"
    if raw is not None:
        return f"{raw:.0%}"
    if "rule:" in src:
        return "Rule"
    return "N/A"


def _infer_action_tag(suggested: str, combined: str) -> str:
    t = f"{suggested} {combined}".lower()
    if any(x in t for x in ("reply", "respond", "rsvp", "confirm attendance", "get back")):
        return "Reply"
    if any(x in t for x in ("schedule", "calendar", "book a", "meeting at", "3:00", "3pm")):
        return "Schedule"
    if any(x in t for x in ("submit", "upload", "attach", "send your", "application")):
        return "Submit"
    if any(x in t for x in ("register", "registration", "sign up")):
        return "Register"
    if any(x in t for x in ("forward", "share with", "fyi to")):
        return "Forward"
    return "Review"


def _parse_deadline_date(deadline_str: str):
    if not deadline_str or not isinstance(deadline_str, str):
        return None
    s = deadline_str.strip()
    if not dateparser:
        return None
    try:
        d = dateparser.parse(
            s,
            settings={
                "PREFER_DATES_FROM": "future",
                "RELATIVE_BASE": datetime.now(),
            },
        )
        return d
    except (TypeError, ValueError, OverflowError):
        return None


def _deadline_bar_style(days_left):
    """Returns (fill_pct 0-100, bar_color, label_suffix)."""
    if days_left is None:
        return None
    if days_left < 0:
        return 100, "#991b1b", "overdue"
    # Closer = fuller bar + redder (more urgent)
    urgency = max(0.0, min(1.0, 1.0 - (days_left / 30.0)))
    fill = max(8, int(20 + urgency * 80))
    # green -> red
    r = int(34 + urgency * (220 - 34))
    g = int(197 - urgency * 150)
    b = int(94 - urgency * 70)
    color = f"rgb({r},{g},{b})"
    if days_left == 0:
        suffix = "today"
    elif days_left == 1:
        suffix = "1 day left"
    else:
        suffix = f"{int(days_left)} days left"
    return fill, color, suffix


def _collect_entity_chips(entities: dict, key_details: dict) -> list[tuple[str, str, str]]:
    """(chip_id, label, value) for styling — order: deadline, org, event, action, rest."""
    chips = []
    seen = set()

    def add(cid, label, val):
        if not val or not str(val).strip():
            return
        key = (cid, str(val).strip()[:200])
        if key in seen:
            return
        seen.add(key)
        chips.append((cid, label, str(val).strip()[:200]))

    kd = key_details or {}
    if kd.get("Deadline"):
        add("deadline", "deadline", kd["Deadline"])
    if kd.get("Organization"):
        add("org", "org", kd["Organization"])
    if kd.get("Event name"):
        add("event", "event", kd["Event name"])
    if kd.get("Required action"):
        add("action", "action", kd["Required action"])

    for val in entities.get("Dates") or []:
        add("deadline", "deadline", val)
    for val in entities.get("Organizations") or []:
        add("org", "org", val)
    for val in entities.get("Events") or []:
        add("event", "event", val)
    for val in entities.get("Money amounts") or []:
        add("money", "money", val)
    for val in entities.get("Times") or []:
        add("time", "time", val)
    for val in entities.get("People involved") or []:
        add("person", "person", val)
    for val in entities.get("Places") or []:
        add("place", "place", val)
    if kd.get("Requirement"):
        add("req", "requirement", kd["Requirement"])

    return chips


CHIP_DOT = {
    "deadline": "#ef4444",
    "org": "#3b82f6",
    "event": "#14b8a6",
    "action": "#f59e0b",
    "money": "#eab308",
    "time": "#a855f7",
    "person": "#ec4899",
    "place": "#22c55e",
    "req": "#64748b",
}


# ---------- CSS ----------
st.markdown(
    """
<style>
body { background-color: #0b0f19; }
.block-container { padding-top: 1.5rem; max-width: 920px; margin: auto; }
/* Title uses native Streamlit widgets; keep results card styling only */
.card {
    background: #151a28; padding: 22px 24px; border-radius: 14px; margin-top: 16px;
    border: 1px solid #1e293b;
}
textarea, input[type="text"] { background-color: #1c2233 !important; color: #f1f5f9 !important;
    border-radius: 10px !important; border: 1px solid #334155 !important; }
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; margin: 16px 0 8px 0; }
.metric-card {
    flex: 1; min-width: 140px; max-width: 280px; padding: 14px 16px; border-radius: 12px;
    background: #0f1419; border: 2px solid #334155;
}
.metric-card.active { box-shadow: 0 0 0 1px rgba(255,255,255,0.06); }
.metric-card.importance-active { border-color: #ef4444; }
.metric-card.priority-high { border-color: #ef4444; }
.metric-card.priority-medium { border-color: #f59e0b; }
.metric-card.priority-low { border-color: #22c55e; opacity: 0.85; }
.metric-card.conf-active { border-color: #eab308; }
.metric-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; margin-bottom: 6px; }
.metric-value { font-size: 22px; font-weight: 700; color: #f8fafc; }
.chip-wrap { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
.entity-chip {
    display: inline-flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: 999px;
    background: #1e293b; border: 1px solid #334155; font-size: 13px; color: #e2e8f0;
}
.chip-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.chip-cat { color: #94a3b8; font-size: 11px; text-transform: lowercase; margin-right: 4px; }
.deadline-track {
    margin-top: 12px; padding: 14px 16px; border-radius: 12px; background: #0f1419;
    border: 1px solid #334155;
}
.deadline-bar-bg { height: 10px; border-radius: 6px; background: #334155; overflow: hidden; margin-top: 10px; }
.deadline-bar-fill { height: 100%; border-radius: 6px; transition: width 0.4s ease; }
.callout {
    margin-top: 18px; padding: 16px 18px; border-radius: 12px;
    background: linear-gradient(135deg, #1e3a5f 0%, #172554 100%);
    border: 1px solid #2563eb;
}
.callout-tag { font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: #93c5fd; margin-bottom: 8px; }
.callout-text { font-size: 17px; font-weight: 600; color: #f8fafc; line-height: 1.45; }
.hint-muted { color: #64748b; font-size: 13px; margin-top: 6px; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [aria-selected="true"] { background: #1e293b !important; border-radius: 8px 8px 0 0; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- HEADER (native widgets — HTML title was stripped by Streamlit sanitizer in some builds) ----------
st.title("Email Overload Organizer")
st.caption(
    "Priority, entities, and deadlines at a glance — not buried in paragraphs."
)

# ---------- INPUT (no HTML wrapper — orphan <div> rendered as an empty box above widgets) ----------
st.divider()
st.subheader("Paste your email")

subject_input = st.text_input("Subject (optional)", key="subj", placeholder="e.g. Re: report — need your section")

body_input = st.text_area(
    "Body",
    key="body",
    height=180,
    placeholder="Paste your email body here.",
)

combined = format_for_classifier(subject_input, body_input)
word_n = len(combined.split()) if combined.strip() else 0
char_n = len(combined)
st.caption(f"**{word_n}** words · **{char_n}** characters — long threads are fine; the model uses subject + body together.")
st.caption(
    "Using **Subject:** and **Body:** lines helps match training format. Leave subject empty for a single block."
)

analyze = st.button("Analyze email", use_container_width=True, disabled=not combined.strip())
if not combined.strip():
    st.markdown('<p class="hint-muted">Add a subject or body to enable analysis.</p>', unsafe_allow_html=True)

# ---------- RESULTS ----------
if analyze and combined.strip():
    with st.spinner("Analyzing..."):
        result = analyzer.analyze(combined)

    classification = result.get("classification", "Low")
    importance = result.get("importance", "Not Important")
    explanation = result.get("explanation", "")
    key_details = result.get("key_details", {}) or {}
    suggested_action = result.get("suggested_action", "")
    entities = result.get("entities", {}) or {}
    conf_text = _confidence_display(result)
    imp_short = "Yes" if importance == "Important" else "No"

    # --- Strategy 1: metric triad ---
    imp_cls = "metric-card importance-active active" if importance == "Important" else "metric-card"
    if classification == "High":
        pri_cls = "metric-card priority-high active"
    elif classification == "Medium":
        pri_cls = "metric-card priority-medium active"
    else:
        pri_cls = "metric-card priority-low active"

    conf_cls = "metric-card conf-active active"
    if conf_text in ("N/A", "Rule"):
        conf_cls = "metric-card"

    triad_html = f"""
<div class="metric-row">
  <div class="{imp_cls}"><div class="metric-label">Importance</div><div class="metric-value">{imp_short}</div></div>
  <div class="{pri_cls}"><div class="metric-label">Priority</div><div class="metric-value">{classification}</div></div>
  <div class="{conf_cls}"><div class="metric-label">Confidence</div><div class="metric-value">{conf_text}</div></div>
</div>
"""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(triad_html, unsafe_allow_html=True)

    # --- Strategy 3: deadline bar ---
    dl_raw = key_details.get("Deadline")
    dl_dt = _parse_deadline_date(dl_raw) if dl_raw else None
    if dl_dt:
        delta = dl_dt.date() - datetime.now().date()
        days_left = delta.days
        bar = _deadline_bar_style(float(days_left))
        if bar:
            fill_pct, bar_color, suffix = bar
            pretty = dl_dt.strftime("%b %d, %Y")
            st.markdown(
                f"""
<div class="deadline-track">
  <div style="display:flex;justify-content:space-between;align-items:center;font-size:13px;color:#94a3b8;">
    <span>Today</span>
    <span style="color:#fca5a5;font-weight:600;">{pretty} — {suffix}</span>
  </div>
  <div class="deadline-bar-bg"><div class="deadline-bar-fill" style="width:{fill_pct}%;background:{bar_color};"></div></div>
</div>
""",
                unsafe_allow_html=True,
            )

    # --- Strategy 2: entity chips ---
    chips = _collect_entity_chips(entities, key_details)
    if chips:
        st.markdown("**Extracted signals**")
        parts = ['<div class="chip-wrap">']
        for cid, label, val in chips:
            dot = CHIP_DOT.get(cid, "#94a3b8")
            tip = html.escape(f"{label}: {val}"[:120])
            parts.append(
                f'<span class="entity-chip" title="{tip}"><span class="chip-dot" style="background:{dot};"></span>'
                f'<span class="chip-cat">{label}</span><span>{html.escape(val)}</span></span>'
            )
        parts.append("</div>")
        st.markdown("".join(parts), unsafe_allow_html=True)

    # --- Strategy 4: tabs (trust / transparency) ---
    sig_rows = result.get("signals") or []
    tab_sum, tab_sig, tab_ent = st.tabs(["Summary", "Signals", "Entities"])

    with tab_sum:
        st.write(explanation)
        if result.get("summary"):
            st.caption("**One-line scan**")
            st.write(result["summary"])

    with tab_sig:
        st.caption("Rules, model scores, and keyword cues that influenced importance.")
        if not sig_rows:
            st.info("No extra signal rows (rule-only path with no keyword list).")
        else:
            for label, detail in sig_rows:
                st.markdown(f"**{label}:** {detail}")

    with tab_ent:
        st.caption("Raw NER buckets from the analyzer.")
        if entities:
            st.json(entities)
        else:
            st.info("No entities (not important — NER skipped).")

    # --- Strategy 5: action callout (final CTA) ---
    tag = _infer_action_tag(suggested_action, combined)
    safe_action = html.escape(suggested_action)
    st.markdown(
        f"""
<div class="callout">
  <div class="callout-tag">Suggested action · {html.escape(tag)}</div>
  <div class="callout-text">{safe_action}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

elif analyze:
    st.warning("Enter at least a subject or body before analyzing.")
