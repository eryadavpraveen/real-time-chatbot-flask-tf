import json
import numpy as np
import nltk
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

from utils import tokenize, lemmatize_words, bag_of_words

# --- NLTK Safe Downloader ---
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")


# --- Load intents.json safely ---
with open("intents.json", "r", encoding="utf-8") as f:
    try:
        intents = json.load(f)
    except json.JSONDecodeError:
        raise Exception("❌ ERROR: intents.json is empty or invalid. Please fix the file.")


words = []
classes = []
documents = []

# Build vocabulary
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        words.extend(w)
        documents.append((w, intent["tag"]))
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

words = lemmatize_words(words)
words = sorted(set(words))
classes = sorted(set(classes))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = bag_of_words(doc[0], words)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

training = np.array(training, dtype=object)
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# --- Model ---
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# --- Train ---
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# --- Save model ---
model.save("chatbot_model.h5", hist)

print("✅ Model training complete. Saved as chatbot_model.h5")
