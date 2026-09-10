"""
Task 4 - Machine Learning Model Implementation
--------------------------------------------------
Builds a predictive text-classification model with scikit-learn to
detect spam vs. ham (legitimate) SMS/email-style messages.

Pipeline: TF-IDF vectorization -> Multinomial Naive Bayes
Also trains a Logistic Regression baseline for comparison.

Usage:
    python spam_classifier.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Dataset
#    A small labeled dataset of SMS/email style messages (spam=1, ham=0).
# ---------------------------------------------------------------------------
DATA = [
    ("Congratulations! You've won a $1000 Walmart gift card. Click here to claim now!!!", 1),
    ("URGENT: Your account has been suspended. Verify your details immediately at this link.", 1),
    ("You have been selected for a FREE iPhone 15! Claim your prize before it expires.", 1),
    ("WINNER!! As a valued customer you have been selected to receive a cash reward.", 1),
    ("Get cheap loans approved instantly, no credit check required! Apply now.", 1),
    ("Limited time offer: buy one get one FREE on all products, click to shop now.", 1),
    ("Your PayPal account needs verification, click the link to avoid suspension.", 1),
    ("Hot singles in your area are waiting to chat with you tonight!", 1),
    ("Claim your lottery winnings of $5,000,000 now by replying with your bank details.", 1),
    ("Congratulations, you have been pre-approved for a $10,000 personal loan.", 1),
    ("FREE entry into our weekly draw, text WIN to 80085 to enter.", 1),
    ("Your package could not be delivered, click here to reschedule and pay a small fee.", 1),
    ("Act now! Your subscription is about to expire, renew today for 90% off.", 1),
    ("You've received a new voicemail, click to listen, verify your number first.", 1),
    ("Earn $500 a day working from home, no experience needed, sign up now!", 1),
    ("Reminder: your dentist appointment is scheduled for tomorrow at 10 AM.", 0),
    ("Hey, are we still meeting for lunch today at the usual place?", 0),
    ("The quarterly sales report has been uploaded to the shared drive, please review.", 0),
    ("Can you send me the notes from yesterday's lecture when you get a chance?", 0),
    ("Happy birthday! Hope you have a wonderful day, let's catch up soon.", 0),
    ("Team meeting has been moved to 3 PM in conference room B.", 0),
    ("Thanks for the help with the project, really appreciate it.", 0),
    ("Please find attached the invoice for last month's services.", 0),
    ("Don't forget to bring your laptop charger to class tomorrow.", 0),
    ("Mom, I'll be home by 8, can you save some dinner for me?", 0),
    ("The internship task submissions are due by end of this week.", 0),
    ("Your Amazon order has been shipped and will arrive on Thursday.", 0),
    ("Let's schedule a call to discuss the project requirements this week.", 0),
    ("Great job on the presentation today, the client was impressed.", 0),
    ("Reminder: library books are due for return by Friday.", 0),
    ("Could you review my code and share feedback before the deadline?", 0),
    ("The weather looks great this weekend, want to go hiking?", 0),
    ("Your electricity bill for this month is ready to view online.", 0),
    ("I've booked the conference room for our 2 PM meeting tomorrow.", 0),
    ("Congrats on completing the certification course successfully!", 0),
    ("Attached is the updated resume, let me know if any changes are needed.", 0),
    ("FINAL NOTICE: pay your outstanding balance now to avoid legal action, click here.", 1),
    ("You are eligible for a government tax refund, submit your details to claim.", 1),
    ("Click this link to unlock exclusive crypto investment returns of 300%.", 1),
    ("Verify your bank OTP now to prevent unauthorized account access.", 1),
]

df = pd.DataFrame(DATA, columns=["message", "label"])


def main():
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.25,
        random_state=RANDOM_STATE, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    }

    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, (name, model) in zip(axes, models.items()):
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        results[name] = dict(accuracy=acc, precision=prec, recall=rec, f1=f1)

        print(f"\n=== {name} ===")
        print(f"Accuracy : {acc:.2f}")
        print(f"Precision: {prec:.2f}")
        print(f"Recall   : {rec:.2f}")
        print(f"F1-score : {f1:.2f}")
        print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"], zero_division=0))

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name)

    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150)
    print("\nConfusion matrix figure saved to confusion_matrices.png")

    # Quick prediction demo on new, unseen messages
    best_model = models["Multinomial Naive Bayes"]
    sample_msgs = [
        "Congratulations! You have won a free cruise, click to claim your prize now!",
        "Hey, can we reschedule our meeting to 4 PM tomorrow?",
    ]
    sample_vec = vectorizer.transform(sample_msgs)
    preds = best_model.predict(sample_vec)
    print("\n=== Sample Predictions (Naive Bayes) ===")
    for msg, pred in zip(sample_msgs, preds):
        label = "SPAM" if pred == 1 else "HAM"
        print(f"[{label}] {msg}")

    results_df = pd.DataFrame(results).T.round(3)
    results_df.to_csv("model_comparison.csv")
    print("\nModel comparison:\n", results_df)


if __name__ == "__main__":
    main()
