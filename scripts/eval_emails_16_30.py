"""
Evaluate EmailClassifier on held-out Enron-style stress emails (16–30).
Does not train — inference only.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.classification import EmailClassifier
from src.email_format import format_for_classifier

# (id, subject, body, true_label) — 1=Important, 0=Not Important
CASES = [
    (
        16,
        "Re: numbers again",
        """Yeah, I noticed that too.

If we don't fix those discrepancies before the call tomorrow, it might raise questions from their side.

Let's align today if possible.""",
        1,
    ),
    (
        17,
        "Fwd: approval",
        """Passing this along — see below.

---

Can you confirm if we're okay to move ahead? We need sign-off before proceeding with the rollout later today.""",
        1,
    ),
    (
        18,
        "FYI",
        """Just looping you in — the vendor mentioned they won't be able to deliver unless we send the updated specs by tonight.""",
        1,
    ),
    (
        19,
        "update on progress",
        """Hi,
Overall things seem to be moving in the right direction. We've managed to close a few pending items and are waiting on confirmations from two teams. There are still some minor concerns, but nothing that should significantly delay the process.

One thing to note — the final submission needs to go out before 6 PM today, so we should wrap everything up accordingly.

Let me know if you see any issues.""",
        1,
    ),
    (
        20,
        "quick favor",
        """Hey,
When you get a chance, can you send over the latest client list? I'll need it before I send out the invites later today.""",
        1,
    ),
    (
        21,
        "urgent update",
        """Don't forget to check out the new features in our internal portal. Deadline to explore them is this week!""",
        0,
    ),
    (
        22,
        "file",
        """I can't open the shared document. Might need access before I can continue.""",
        1,
    ),
    (
        23,
        "discussion",
        """We might have a quick discussion later today about this. Join if you're around.""",
        0,
    ),
    (
        24,
        "small thing",
        """Hey,
No rush, but I'll need the updated draft before the end of the day to send it across.""",
        1,
    ),
    (
        25,
        "update",
        """Client just confirmed they're expecting delivery tomorrow morning.""",
        1,
    ),
    (
        26,
        "status",
        """Hi,
We've been reviewing the current setup and everything seems mostly aligned with expectations. There are still a few details to finalize, but overall progress is steady.

We still need your approval before we can proceed further.

Thanks""",
        1,
    ),
    (
        27,
        "action required",
        """You're receiving this email as part of our mailing list. No action is needed unless you wish to unsubscribe.""",
        0,
    ),
    (
        28,
        "Re: timeline",
        """That timeline works. Just make sure everything is ready before tomorrow's submission.""",
        1,
    ),
    (
        29,
        "quick check",
        """Are you okay with the changes? I'll proceed based on your confirmation.""",
        1,
    ),
    (
        30,
        "heads up",
        """Office parking rules are changing next month.""",
        0,
    ),
]


def main():
    clf = EmailClassifier()
    y_true = []
    y_pred = []
    rows = []

    for eid, subj, body, gold in CASES:
        text = format_for_classifier(subj, body)
        meta = clf.predict_with_meta(text)
        pred = meta["label"]
        y_true.append(gold)
        y_pred.append(pred)
        rows.append((eid, gold, pred, meta.get("source"), meta))

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("=" * 72)
    print("Stress evaluation: emails 16-30 (inference only, not used for training)")
    print("=" * 72)
    print(f"\nSamples: {len(CASES)}  (Important: {sum(y_true)}, Not important: {len(y_true) - sum(y_true)})")
    print(f"\nOverall accuracy: {acc:.4f} ({sum(y == p for y, p in zip(y_true, y_pred))}/{len(y_true)} correct)")
    print(f"Macro F1 (important vs not): {f1:.4f}")
    print("\nConfusion matrix [rows=true, cols=pred]  labels order [0, 1]:")
    print("                pred 0    pred 1")
    print(f"  true 0 (not):    {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"  true 1 (imp):    {cm[1,0]:4d}      {cm[1,1]:4d}")
    print("\n" + classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Not important (0)", "Important (1)"],
        digits=4,
        zero_division=0,
    ))

    print("Per-email (OK / MISS):")
    for eid, gold, pred, src, meta in rows:
        ok = "OK" if gold == pred else "MISS"
        print(f"  {ok}  #{eid:2d}  true={gold}  pred={pred}  source={src}")
        if gold != pred:
            conf = meta.get("adjusted_confidence") or meta.get("confidence")
            conf_s = f"{conf:.3f}" if conf is not None else "n/a"
            print(f"         (confidence/adj: {conf_s})")

    print()


if __name__ == "__main__":
    main()
