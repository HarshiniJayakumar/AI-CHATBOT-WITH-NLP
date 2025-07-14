
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import nltk
import spacy
import os
from nltk.corpus import stopwords
from contextlib import redirect_stdout

# Suppress NLTK downloader print
with open(os.devnull, "w") as f, redirect_stdout(f):
    nltk.download('stopwords', quiet=True)

# Load English spaCy model
nlp = spacy.load("en_core_web_sm")

# Load NLTK stopwords
stop_words = set(stopwords.words('english'))

# Load CSV files
faq_data = pd.read_csv("questions.csv")
symptom_data = pd.read_csv("symptoms.csv")

# Emergency keywords
emergency_keywords = ["emergency", "bleeding", "heart attack", "unconscious", "not breathing"]

# Check for emergency
def check_emergency(user_input):
    for keyword in emergency_keywords:
        if keyword in user_input.lower():
            return True
    return False

# FAQ matching
def match_faq(user_input):
    for _, row in faq_data.iterrows():
        if row['question'].lower() in user_input.lower():
            return row['answer']
    return None

# Symptom matching (supports multiple symptoms)
def match_symptom(user_input):
    # Preprocess input using spaCy
    doc = nlp(user_input.lower())
    tokens = [token.text for token in doc if token.is_alpha and token.text not in stop_words]

    matched_conditions = []
    for _, row in symptom_data.iterrows():
        symptom_keywords = [s.strip().lower() for s in row['symptom_keywords'].split(',')]
        if any(symptom in tokens for symptom in symptom_keywords):
            matched_conditions.append(row['possible_condition'])

    return list(set(matched_conditions))  # Remove duplicates

# Run chatbot
print("🤖 Health Bot: Hello! I can help with health FAQs and symptom checking.")
print("Ask a health question or describe your symptoms.")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("🤖 Health Bot: Take care! Goodbye.")
        break

    if check_emergency(user_input):
        print("🚨 Health Bot: This seems like an emergency. Please call 108 or go to the nearest hospital immediately!")
        continue

    faq_answer = match_faq(user_input)
    if faq_answer:
        print("📘 Health Bot FAQ: " + faq_answer)
        continue

    symptoms = match_symptom(user_input)
    if symptoms:
        print("🩺 Health Bot: Based on your symptoms, possible condition(s) could be:")
        for condition in symptoms:
            print(f"- {condition}")
    else:
        print("🤔 Health Bot: Sorry, I couldn't understand. Please ask a health-related question or describe your symptoms.")


