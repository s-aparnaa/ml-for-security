# phishing_classifier.py

# 🛡️ Phishing URL Classifier using Logistic Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
DATA_PATH = "data/phishing_site_urls.csv"  # Adjust this if your data is elsewhere
df = pd.read_csv(DATA_PATH)

# Check first few rows
print("Dataset loaded. Sample rows:")
print(df.head())

# Clean and rename columns if needed
if 'Label' in df.columns and 'URL' in df.columns:
    df = df.rename(columns={'Label': 'label', 'URL': 'text'})
elif 'text' not in df.columns or 'label' not in df.columns:
    raise ValueError("Dataset must have 'text' and 'label' columns.")

# Convert label to binary
df['label'] = df['label'].map({'phishing': 1, 'legitimate': 0})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate model
y_pred = model.predict(X_test_vec)
print("Classification Report:")
print(classification_report(y_test, y_pred))
