"""Stress test from emails_dataset_ai.pdf — compare stacking classifier to PDF labels.

Run from project root:
  python -m scripts.stress_test_pdf_emails
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classification import EmailClassifier
from src.email_format import format_for_classifier

# Subject + body + true label (1=Important, 0=Not Important) from PDF
CASES = [
    (1, "Quick thing when you get a chance", "Hey, Can you take a look at the document I shared yesterday and make sure everything looks okay? It would be great if this could be done ASAP, but no rush if you're tied up. Thanks!", 1),
    (2, "Slides update", "Hi, Just a small ask — could you update the slides before tomorrow's meeting? Nothing major, just tweak the numbers in section 3. Cheers", 1),
    (3, "LAST CHANCE: Offer expires tonight!", "Hurry! Your exclusive 50% discount expires at midnight. Don't miss out on this limited-time deal — shop now before it's gone forever.", 0),
    (4, "Dinner plan", "Hey, Are we still on for dinner tonight? Let me know by 6 PM so I can make the reservation.", 1),
    (5, "Project discussion follow-up", "Hi, Thanks again for joining the discussion earlier today... Also, please submit your finalized report by EOD today. Let me know if you have any questions.", 1),
    (6, "Small favor", "Hi, Whenever you have a moment, could you send me the client feedback from last week? We might need it for something soon.", 1),
    (7, "Action required immediately", "Your account preferences need to be reviewed immediately to continue receiving newsletters.", 0),
    (8, "Quick confirmation", "Hey, Just wanted to check if you're okay with the changes... I need to confirm before sending it out later today.", 1),
    (9, "Need your help", "Hi, I know things have been hectic... But the submission deadline for the form is tonight.", 1),
    (10, "Idea for later", "Hey, We could explore that feature sometime. No hurry.", 0),
    (11, "Meeting overlap", "Hi, Meeting tomorrow at 3 PM overlaps with another one. Let me know what to do.", 1),
    (12, "Approval needed", "Hi, Please review and approve the document. We can't proceed without it.", 1),
    (13, "Just a random thought", "Hey, There's an issue in authentication. If not fixed before tonight's release, users may face login issues.", 1),
    (14, "Notes from today", "Hi, Discussion summary... Compliance form must be submitted before 5 PM today.", 1),
    (15, "Quick check", "Hey, Need confirmation on pricing before sending to client.", 1),
    (16, "IMPORTANT: Deadline approaching", "Final reminder about membership discount.", 0),
    (17, "Tiny thing", "Hey, Server config still has old API key. Requests may fail tonight.", 1),
    (18, "General update", "Hi, Progress update... Contract must be signed today or deal postponed.", 1),
    (19, "Urgent response needed", "Please complete this survey before deadline.", 0),
    (20, "About the file", "Hi, Need your section before I finish report today.", 1),
    (21, "Action required", "You subscribed to newsletter. No action needed.", 0),
    (22, "Small clarification", "Hey, Confirm venue before I finalize today.", 1),
    (23, "Deadline reminder", "Newsletter submission deadline tomorrow (optional).", 0),
    (24, "Access", "Hi, Access revoked. Can't proceed until restored.", 1),
    (25, "Submission deadline for financial report", "Dear Team, Submit report by 4 PM tomorrow.", 1),
    (26, "Quick thing", "Hey, Upload assignment before midnight.", 1),
    (27, "Whenever possible", "Hi, Send draft today for review.", 1),
    (28, "Project review meeting scheduled", "Dear All, Meeting Monday 10 AM mandatory.", 1),
    (29, "Catch up tomorrow?", "Hey, Let's meet tomorrow at 3 PM.", 1),
    (30, "Optional sync", "Hi, Casual sync Friday. Join if interested.", 0),
    (31, "Invitation to Annual Tech Symposium", "You are invited. Confirm attendance.", 0),
    (32, "Party this weekend n", "Hey, Come if free.", 0),
    (33, "Workshop coordination", "Hi, Lead session Thursday. Share materials by Wednesday.", 1),
    (34, "System maintenance update", "Maintenance Sunday. Services unavailable.", 0),
    (35, "Policy update", "Review policy before next claim.", 1),
    (36, "Just FYI", "Office timings shift next month.", 0),
    (37, "Small reminder", "Hey, Need finalized design by evening.", 1),
    (38, "Webinar invite", "Join if interested.", 0),
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
            wrong.append((idx, subj[:45], y, pred, meta.get("source"), meta.get("confidence")))

    n = len(CASES)
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    print(f"Dataset: {n} emails (from emails_dataset_ai.pdf)")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision (Important): {prec:.4f}")
    print(f"Recall (Important):    {rec:.4f}")
    print(f"F1 (Important):        {f1:.4f}")
    print(f"Confusion: TP={tp} TN={tn} FP={fp} FN={fn}")
    print()
    print(f"Misclassified ({len(wrong)}):")
    for w in wrong:
        print(f"  Email {w[0]:2d} | true={w[2]} pred={w[3]} | {w[4]} | conf={w[5]}")


if __name__ == "__main__":
    main()
