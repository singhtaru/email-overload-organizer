"""Stress test from emails_dataset_ai_enron_style_2.pdf — classifier vs PDF labels."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classification import EmailClassifier
from src.email_format import format_for_classifier

# Subject + body + true label (1=Important, 0=Not Important) from PDF
CASES = [
    (
        1,
        "Re: numbers",
        "Hey, I was going through the figures you sent earlier. Some of the projections don't seem to match what we discussed last week. We're presenting this to senior management tomorrow morning, so it would help to have this cleaned up before then. Let me know.",
        1,
    ),
    (
        2,
        "Fwd: contract draft",
        "FYI — forwarding the latest draft from legal below. -----Original Message----- Please review the attached contract and confirm if we're good to proceed. We're aiming to close this by end of day.",
        1,
    ),
    (
        3,
        "quick one",
        "Can you send me the updated pricing sheet when you get a minute? I might need to share it later today.",
        1,
    ),
    (
        4,
        "update",
        "Hi, Things are moving along as expected. We've made some progress on the vendor side and are waiting on a few confirmations. One thing — we still need your approval on the budget before we can finalize anything. Thanks",
        1,
    ),
    (
        5,
        "meeting",
        "I think I have a conflict at that time. Can we push this to later in the afternoon or sometime tomorrow?",
        1,
    ),
    (
        6,
        "office update",
        "Just a note that the cafeteria hours will be changing starting next week.",
        0,
    ),
    (
        7,
        "Re: file",
        "I didn't see your section in the shared folder. Not sure if I missed it, but I'll need it before I send everything out.",
        1,
    ),
    (
        8,
        "Re: Re: status",
        "We've gone back and forth on this a few times now. I think we're aligned on most points. Only outstanding item is the compliance approval — without that, we can't proceed further. Let me know where that stands.",
        1,
    ),
    (
        9,
        "important update for employees",
        "Don't miss out on our exclusive employee discount program. Limited time offers available now.",
        0,
    ),
    (
        10,
        "access",
        "Looks like my access to the system has been removed. I can't run the reports until this is fixed.",
        1,
    ),
    (
        11,
        "Re: confirmation",
        "Just checking if you're okay with the final numbers. I'll go ahead and send them across if I don't hear otherwise.",
        1,
    ),
    (
        12,
        "team sync",
        "We're having a quick sync later today if anyone wants to join and discuss ideas.",
        0,
    ),
    (
        13,
        "Re: draft",
        "The draft looks fine overall. Just make sure the numbers are updated before you send it out today.",
        1,
    ),
    (
        14,
        "FYI",
        "Just heard back from the client — they're expecting the revised proposal by tomorrow morning.",
        1,
    ),
    (
        15,
        "random",
        "Saw this article and thought it might interest you. No action needed.",
        0,
    ),
]


def main():
    clf = EmailClassifier()
    tp = tn = fp = fn = 0
    wrong = []
    for idx, subj, body, y in CASES:
        text = format_for_classifier(subj, body)
        meta = clf.predict_with_meta(text)
        pred = meta["label"]
        if pred == y:
            if y == 1:
                tp += 1
            else:
                tn += 1
        else:
            if y == 1:
                fn += 1
            else:
                fp += 1
            wrong.append((idx, subj[:40], y, pred, meta.get("source"), meta.get("confidence")))

    n = len(CASES)
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    print("Dataset: emails_dataset_ai_enron_style_2.pdf (15 emails)")
    print(f"Accuracy:  {acc:.4f} ({tp + tn}/{n})")
    print(f"Precision (Important): {prec:.4f}")
    print(f"Recall (Important):    {rec:.4f}")
    print(f"F1 (Important):        {f1:.4f}")
    print(f"Confusion: TP={tp} TN={tn} FP={fp} FN={fn}")
    print()
    if wrong:
        print(f"Misclassified ({len(wrong)}):")
        for w in wrong:
            print(f"  Email {w[0]:2d} | true={w[2]} pred={w[3]} | {w[4]} | conf={w[5]}")
    else:
        print("No misclassifications.")


if __name__ == "__main__":
    main()
