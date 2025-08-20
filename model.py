import random
import json
import numpy as np
import nltk
from tensorflow.keras.models import load_model
from utils import tokenize, lemmatize_words, bag_of_words

# Load intents and trained model
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

model = load_model("chatbot_model.h5")

# Load vocabulary + classes (saved during training)
# If you saved them to files, load here. Otherwise, rebuild:
words = []
classes = []
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        words.extend(w)
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

words = sorted(set(lemmatize_words(words)))
classes = sorted(set(classes))


def predict_class(sentence):
    """Convert sentence into bag of words, predict intent class"""
    bow = bag_of_words(tokenize(sentence), words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    """Pick a random response from predicted intent"""
    if len(intents_list) == 0:
        return "Sorry, I didn’t quite get that."

    tag = intents_list[0]["intent"]
    for i in intents_json["intents"]:
        if i["tag"] == tag:
            return random.choice(i["responses"])
    return "I'm not sure how to respond to that."
