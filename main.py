"""
Spam Filter — TF-IDF + Naive Bayes with real-time detection
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib, re

# Sample training data (spam + ham messages)
TRAINING_DATA = [
    ("Get FREE money now!!! Click here to claim your prize!", "spam"),
    ("Win $1000 GUARANTEED! Call now!", "spam"),
    ("URGENT: Your account will be suspended! Verify now!", "spam"),
    ("You have been selected for a special offer!", "spam"),
    ("Congrats! You won an iPhone! Click to claim!", "spam"),
    ("FREE ENTRY: Text WIN to 80488 to win a Nokia", "spam"),
    ("Dear customer, your loan is approved! Apply now!", "spam"),
    ("Hot singles in your area! Click to meet them!", "spam"),
    ("Hi, are you coming to the meeting tomorrow?", "ham"),
    ("Can you pick up some milk on your way home?", "ham"),
    ("The project deadline has been moved to Friday.", "ham"),
    ("Lunch at noon? Let me know if that works.", "ham"),
    ("Please review the attached report and share feedback.", "ham"),
    ("Your package has been shipped. Tracking: AB123456", "ham"),
    ("Reminder: Doctor appointment at 3pm on Thursday.", "ham"),
    ("Great job on the presentation! The client loved it.", "ham"),
    ("Can we reschedule our call to next week?", "ham"),
    ("The meeting notes have been shared with the team.", "ham"),
]

def generate_more_data():
    """Generate synthetic spam/ham data."""
    spam_patterns = [
        "Win {amount} now! Click {url}",
        "FREE {item}! Limited time offer!",
        "URGENT: Your account needs verification!",
        "Congratulations! You have been selected!",
        "Make ${amount} from home! Easy money!",
    ]
    ham_patterns = [
        "Hi, just checking in about {topic}.",
        "The meeting is scheduled for {time}.",
        "Please review {document} at your convenience.",
        "Following up on our conversation about {topic}.",
        "The report for {topic} is ready for review.",
    ]
    data = list(TRAINING_DATA)
    amounts = ["$1000", "$500", "£200", "€5000"]
    items = ["iPhone", "iPad", "laptop", "gift card"]
    topics = ["the project", "the budget", "the report", "the proposal"]
    for _ in range(200):
        pat = np.random.choice(spam_patterns)
        text = pat.format(amount=np.random.choice(amounts), url="www.click-here.com",
                          item=np.random.choice(items))
        data.append((text, "spam"))
    for _ in range(200):
        pat = np.random.choice(ham_patterns)
        text = pat.format(topic=np.random.choice(topics), time="3pm tomorrow",
                          document="the Q3 report")
        data.append((text, "ham"))
    return pd.DataFrame(data, columns=["text", "label"])

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "URL", text)
    text = re.sub(r"\d+", "NUM", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return text.strip()

def train(df):
    df["text_clean"] = df["text"].apply(clean_text)
    X = df["text_clean"]
    y = (df["label"] == "spam").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
            ("clf", MultinomialNB(alpha=0.1))
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
            ("clf", LogisticRegression(max_iter=1000, C=5.0))
        ]),
    }

    best_model, best_f1 = None, 0
    for name, model in models.items():
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
        print(f"  {name}: F1={cv.mean():.4f}")
        if cv.mean() > best_f1:
            best_f1 = cv.mean()
            best_model = model

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    print(f"\nTest Results:")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

    joblib.dump(best_model, "spam_filter.pkl")
    print("Model saved: spam_filter.pkl")
    return best_model

def classify(model, text):
    prob = model.predict_proba([clean_text(text)])[0]
    label = "SPAM" if prob[1] > 0.5 else "HAM"
    print(f"  [{label}] ({prob[1]:.1%} spam probability): {text[:60]}...")
    return label

def main():
    print("=" * 60)
    print("  Spam Filter (TF-IDF + Naive Bayes/Logistic Regression)")
    print("=" * 60)
    df = generate_more_data()
    print(f"Dataset: {len(df)} messages ({df.label.value_counts().to_dict()})")
    print("\nTraining classifier...")
    model = train(df)

    print("\nReal-time classification:")
    test_messages = [
        "WIN a brand new iPhone! Click here now!!!",
        "Hi, the meeting is moved to 3pm on Thursday.",
        "Congratulations! You have won $5000!",
        "Please review the attached quarterly report.",
        "URGENT: Your bank account requires immediate verification!",
    ]
    for msg in test_messages:
        classify(model, msg)

if __name__ == "__main__":
    main()
