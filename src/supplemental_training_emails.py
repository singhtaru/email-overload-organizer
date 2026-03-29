"""
Hand-labeled benchmark emails (PDF stress tests) for optional merge into stacking training.

Labels: 1 = Important, 0 = Not Important.
Texts use Subject:/Body: format (see email_format).
"""

from src.email_format import format_for_classifier

# (subject, body, label) — merged from emails_dataset_ai.pdf + emails_dataset_ai_enron_style_2.pdf
_RAW = [
    ("Quick thing when you get a chance", "Hey, Can you take a look at the document I shared yesterday and make sure everything looks okay? It would be great if this could be done ASAP, but no rush if you're tied up. Thanks!", 1),
    ("Slides update", "Hi, Just a small ask — could you update the slides before tomorrow's meeting? Nothing major, just tweak the numbers in section 3. Cheers", 1),
    ("LAST CHANCE: Offer expires tonight!", "Hurry! Your exclusive 50% discount expires at midnight. Don't miss out on this limited-time deal — shop now before it's gone forever.", 0),
    ("Dinner plan", "Hey, Are we still on for dinner tonight? Let me know by 6 PM so I can make the reservation.", 1),
    ("Project discussion follow-up", "Hi, Thanks again for joining the discussion earlier today... Also, please submit your finalized report by EOD today. Let me know if you have any questions.", 1),
    ("Small favor", "Hi, Whenever you have a moment, could you send me the client feedback from last week? We might need it for something soon.", 1),
    ("Action required immediately", "Your account preferences need to be reviewed immediately to continue receiving newsletters.", 0),
    ("Quick confirmation", "Hey, Just wanted to check if you're okay with the changes... I need to confirm before sending it out later today.", 1),
    ("Need your help", "Hi, I know things have been hectic... But the submission deadline for the form is tonight.", 1),
    ("Idea for later", "Hey, We could explore that feature sometime. No hurry.", 0),
    ("Meeting overlap", "Hi, Meeting tomorrow at 3 PM overlaps with another one. Let me know what to do.", 1),
    ("Approval needed", "Hi, Please review and approve the document. We can't proceed without it.", 1),
    ("Just a random thought", "Hey, There's an issue in authentication. If not fixed before tonight's release, users may face login issues.", 1),
    ("Notes from today", "Hi, Discussion summary... Compliance form must be submitted before 5 PM today.", 1),
    ("Quick check", "Hey, Need confirmation on pricing before sending to client.", 1),
    ("IMPORTANT: Deadline approaching", "Final reminder about membership discount.", 0),
    ("Tiny thing", "Hey, Server config still has old API key. Requests may fail tonight.", 1),
    ("General update", "Hi, Progress update... Contract must be signed today or deal postponed.", 1),
    ("Urgent response needed", "Please complete this survey before deadline.", 0),
    ("About the file", "Hi, Need your section before I finish report today.", 1),
    ("Action required", "You subscribed to newsletter. No action needed.", 0),
    ("Small clarification", "Hey, Confirm venue before I finalize today.", 1),
    ("Deadline reminder", "Newsletter submission deadline tomorrow (optional).", 0),
    ("Access", "Hi, Access revoked. Can't proceed until restored.", 1),
    ("Submission deadline for financial report", "Dear Team, Submit report by 4 PM tomorrow.", 1),
    ("Quick thing", "Hey, Upload assignment before midnight.", 1),
    ("Whenever possible", "Hi, Send draft today for review.", 1),
    ("Project review meeting scheduled", "Dear All, Meeting Monday 10 AM mandatory.", 1),
    ("Catch up tomorrow?", "Hey, Let's meet tomorrow at 3 PM.", 1),
    ("Optional sync", "Hi, Casual sync Friday. Join if interested.", 0),
    ("Invitation to Annual Tech Symposium", "You are invited. Confirm attendance.", 0),
    ("Party this weekend n", "Hey, Come if free.", 0),
    ("Workshop coordination", "Hi, Lead session Thursday. Share materials by Wednesday.", 1),
    ("System maintenance update", "Maintenance Sunday. Services unavailable.", 0),
    ("Policy update", "Review policy before next claim.", 1),
    ("Just FYI", "Office timings shift next month.", 0),
    ("Small reminder", "Hey, Need finalized design by evening.", 1),
    ("Webinar invite", "Join if interested.", 0),
    ("Re: numbers", "Hey, I was going through the figures you sent earlier. Some of the projections don't seem to match what we discussed last week. We're presenting this to senior management tomorrow morning, so it would help to have this cleaned up before then. Let me know.", 1),
    ("Fwd: contract draft", "FYI — forwarding the latest draft from legal below. -----Original Message----- Please review the attached contract and confirm if we're good to proceed. We're aiming to close this by end of day.", 1),
    ("quick one", "Can you send me the updated pricing sheet when you get a minute? I might need to share it later today.", 1),
    ("update", "Hi, Things are moving along as expected. We've made some progress on the vendor side and are waiting on a few confirmations. One thing — we still need your approval on the budget before we can finalize anything. Thanks", 1),
    ("meeting", "I think I have a conflict at that time. Can we push this to later in the afternoon or sometime tomorrow?", 1),
    ("office update", "Just a note that the cafeteria hours will be changing starting next week.", 0),
    ("Re: file", "I didn't see your section in the shared folder. Not sure if I missed it, but I'll need it before I send everything out.", 1),
    ("Re: Re: status", "We've gone back and forth on this a few times now. I think we're aligned on most points. Only outstanding item is the compliance approval — without that, we can't proceed further. Let me know where that stands.", 1),
    ("important update for employees", "Don't miss out on our exclusive employee discount program. Limited time offers available now.", 0),
    ("access", "Looks like my access to the system has been removed. I can't run the reports until this is fixed.", 1),
    ("Re: confirmation", "Just checking if you're okay with the final numbers. I'll go ahead and send them across if I don't hear otherwise.", 1),
    ("team sync", "We're having a quick sync later today if anyone wants to join and discuss ideas.", 0),
    ("Re: draft", "The draft looks fine overall. Just make sure the numbers are updated before you send it out today.", 1),
    ("FYI", "Just heard back from the client — they're expecting the revised proposal by tomorrow morning.", 1),
    ("random", "Saw this article and thought it might interest you. No action needed.", 0),
]

# Repeat supplemental rows so they influence a large Enron batch during retraining.
SUPPLEMENTAL_DUPLICATES = 8

SUPPLEMENTAL_TEXTS_LABELS = []
for _ in range(SUPPLEMENTAL_DUPLICATES):
    for subj, body, y in _RAW:
        SUPPLEMENTAL_TEXTS_LABELS.append((format_for_classifier(subj, body), y))
