import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MODEL_PATH = "models/model.pkl"
VECT_PATH = "models/vectorizer.pkl"


# Topics
TOPICS = [
    "Artificial Intelligence",
    "Machine Learning",
    "DBMS",
    "Operating Systems",
    "Computer Networks",
    "Data Structures",
    "Software Engineering",
    "Computer Architecture",
    "Cyber Security"
]


# Keywords for weak supervision
KEYWORDS = {
    "Artificial Intelligence": [
        "ai", "intelligence", "agent", "search", "heuristic"
    ],

    "Machine Learning": [
        "learning", "model", "training", "classification", "regression"
    ],

    "DBMS": [
        "database", "sql", "table", "query",
        "normalization", "transaction"
    ],

    "Operating Systems": [
        "process", "memory", "os", "scheduling",
        "thread", "deadlock"
    ],

    "Computer Networks": [
        "network", "protocol", "tcp", "ip",
        "routing", "osi"
    ],

    "Data Structures": [
        "tree", "graph", "stack", "queue",
        "linked list", "hashing"
    ],

    "Software Engineering": [
        "software", "testing", "agile",
        "design", "development"
    ],

    "Computer Architecture": [
        "cpu", "cache", "pipeline",
        "instruction", "register"
    ],

    "Cyber Security": [
        "security", "encryption", "attack",
        "firewall", "cryptography"
    ]
}


# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)


# Logistic Regression Model
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# ---------------------------------
# AUTO LABEL USING KEYWORDS
# ---------------------------------

def auto_label(text):
    """
    Assigns a topic label based on keyword frequency.
    Used for weak supervision.
    """

    text = text.lower()

    scores = []

    for topic in TOPICS:

        score = sum(
            text.count(keyword)
            for keyword in KEYWORDS[topic]
        )

        scores.append(score)

    # If no topic keywords are found,
    # default to Artificial Intelligence
    if max(scores) == 0:
        return 0

    return scores.index(max(scores))


# ---------------------------------
# TRAIN MODEL
# ---------------------------------

def train_model(texts):

    os.makedirs("models", exist_ok=True)

    # Remove empty or invalid texts
    texts = [
        text.strip()
        for text in texts
        if isinstance(text, str) and text.strip()
    ]

    if len(texts) < 2:
        print("⚠️ Not enough data to train the model.")
        return False

    # Generate weakly supervised labels
    labels = [
        auto_label(text)
        for text in texts
    ]

    # At least two classes are required
    if len(set(labels)) < 2:
        print(
            "⚠️ Only one topic class detected. "
            "More diverse question papers are required."
        )
        return False

    try:

        # Convert text to TF-IDF features
        X = vectorizer.fit_transform(texts)

        # Train Logistic Regression model
        model.fit(X, labels)

        # Save model and vectorizer
        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECT_PATH)

        print("✅ Model trained successfully.")

        print(
            "Detected classes:",
            [
                TOPICS[class_id]
                for class_id in model.classes_
            ]
        )

        return True

    except Exception as error:

        print("❌ Training error:", error)

        return False


# ---------------------------------
# LOAD MODEL
# ---------------------------------

def load_model():

    if (
        os.path.exists(MODEL_PATH)
        and os.path.exists(VECT_PATH)
    ):

        loaded_model = joblib.load(MODEL_PATH)

        loaded_vectorizer = joblib.load(VECT_PATH)

        return loaded_model, loaded_vectorizer

    return None, None


# ---------------------------------
# PREDICT TOPICS
# ---------------------------------

def predict(text):

    model_loaded, vectorizer_loaded = load_model()

    if model_loaded is None:

        print("⚠️ Model files not found.")

        return []

    if not isinstance(text, str) or not text.strip():

        return []

    try:

        # Convert input text using
        # the trained TF-IDF vectorizer
        X = vectorizer_loaded.transform([text])

        # Get prediction probabilities
        probabilities = model_loaded.predict_proba(X)[0]

        # Get actual classes learned by the model
        classes = model_loaded.classes_

        # Correctly map class IDs to topics
        results = [

            (
                TOPICS[class_id],
                float(probability)
            )

            for class_id, probability
            in zip(classes, probabilities)

        ]

        # Sort predictions by confidence
        results.sort(
            key=lambda item: item[1],
            reverse=True
        )

        return results

    except Exception as error:

        print("❌ Prediction error:", error)

        return []
