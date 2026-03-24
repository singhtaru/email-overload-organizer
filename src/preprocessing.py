from pipeline import EmailAnalyzer

analyzer = EmailAnalyzer()

email = """
Hi John, the meeting with Enron is tomorrow at 10 AM.
"""

print(analyzer.analyze(email))