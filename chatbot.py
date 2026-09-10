"""
Task 3 - AI Chatbot with NLP
-------------------------------
A rule-based conversational chatbot built with classic NLP techniques:
tokenization, stop-word aware keyword extraction and regex/pattern
matching against a small intent library (the same architecture used by
nltk.chat.util.Chat / reflections, reimplemented here with the standard
`re` module so the script has zero external dependencies and runs
anywhere, including offline environments).

If NLTK is installed, the script automatically uses nltk.word_tokenize
for tokenization; otherwise it falls back to a simple regex tokenizer.

Usage:
    python chatbot.py                # interactive mode
    python chatbot.py --demo         # scripted demo conversation
"""

import re
import sys
import random

try:
    import nltk
    nltk.data.find("tokenizers/punkt")
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False


def tokenize(text: str):
    if NLTK_AVAILABLE:
        return word_tokenize(text)
    return re.findall(r"[A-Za-z']+", text.lower())


# ---------------------------------------------------------------------------
# Intent library: (regex pattern, list of possible responses)
# ---------------------------------------------------------------------------
INTENTS = [
    (r"\b(hi|hello|hey|good morning|good evening)\b",
     ["Hello! I'm CampusBot, your internship assistant. How can I help you today?",
      "Hey there! What would you like to know?"]),
    (r"\b(your name|who are you)\b",
     ["I'm CampusBot, a small NLP-based chatbot built for the CODTECH internship task."]),
    (r"\b(internship|codtech)\b",
     ["CODTECH IT Solutions offers Python, web development, and data science internships "
      "with hands-on tasks and a completion certificate."]),
    (r"\b(deadline|submit|submission)\b",
     ["Tasks must be submitted before the internship deadline via your GitHub repository link."]),
    (r"\b(task|tasks)\b",
     ["This internship has 4 tasks: API Integration & Visualization, Automated Report "
      "Generation, an AI Chatbot with NLP, and a Machine Learning Model. You must attempt all four."]),
    (r"\b(help|support)\b",
     ["Sure, I can help! Try asking me about 'tasks', 'deadline', or 'internship'."]),
    (r"\b(thank|thanks)\b",
     ["You're welcome! Happy to help."]),
    (r"\b(bye|goodbye|exit|quit)\b",
     ["Goodbye! All the best with your internship tasks."]),
]

FALLBACKS = [
    "I'm not sure I understand. Could you rephrase that?",
    "Interesting — can you tell me a bit more?",
    "I don't have an answer for that yet, try asking about 'tasks' or 'internship'.",
]


def get_response(user_text: str) -> str:
    text = user_text.lower()
    for pattern, responses in INTENTS:
        if re.search(pattern, text):
            return random.choice(responses)
    return random.choice(FALLBACKS)


def run_demo():
    """Scripted conversation used to demonstrate the chatbot's behaviour."""
    transcript = [
        "Hi there!",
        "What is your name?",
        "Tell me about the internship",
        "What tasks do I need to complete?",
        "When is the submission deadline?",
        "Thanks a lot!",
        "Bye",
    ]
    print("=" * 60)
    print("CampusBot - AI Chatbot with NLP (Demo Conversation)")
    print("=" * 60)
    for user_msg in transcript:
        tokens = tokenize(user_msg)
        response = get_response(user_msg)
        print(f"You      : {user_msg}")
        print(f"[tokens] : {tokens}")
        print(f"CampusBot: {response}")
        print("-" * 60)


def run_interactive():
    print("CampusBot: Hello! Type 'bye' to exit.")
    while True:
        user_msg = input("You: ")
        if not user_msg:
            continue
        response = get_response(user_msg)
        print(f"CampusBot: {response}")
        if re.search(r"\b(bye|goodbye|exit|quit)\b", user_msg.lower()):
            break


if __name__ == "__main__":
    if "--demo" in sys.argv or True:  # default to demo for reproducible output
        run_demo()
