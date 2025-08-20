import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer

# Ensure nltk resources are available
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)

lemmatizer = WordNetLemmatizer()

def tokenize(sentence):
    """Tokenize a sentence into words (string input only)."""
    if isinstance(sentence, list):  # Already tokenized
        return sentence
    return nltk.word_tokenize(str(sentence))

def lemmatize_words(words):
    """Lemmatize each word."""
    return [lemmatizer.lemmatize(w.lower()) for w in words]

def bag_of_words(sentence, words):
    """
    Return bag of words array: 1 for each known word that exists in the sentence.
    sentence -> string
    words -> vocabulary (list of words)
    """
    # Ensure sentence is tokenized correctly
    sentence_words = lemmatize_words(tokenize(sentence))
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1.0
    return bag
