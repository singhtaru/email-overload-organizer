from src.classification import EmailClassifier

def main():
    print("Loading classifier...")
    classifier = EmailClassifier()
    
    text = "Tech talk / Internship Registration - 2027 Batch Date of Visit: 25.03.2026 Eligible Branches B.Tech (CSE / IT related branches only) Eligibility Criteria % in X and XII – 80% or 8.0 CGPA Stipend Will be announced later Last date for Registration 24.03.2026"
    
    print("Predicting...")
    pred = classifier.predict(text)
    print(f"Prediction for University Announcement: {pred}")
    print("If 1, the model successfully generalized to semantic urgency instead of just Enron keywords!")

if __name__ == '__main__':
    main()
