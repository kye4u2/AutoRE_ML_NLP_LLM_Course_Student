import json
import string

import numpy as np
from keras.src.preprocessing.text import Tokenizer, tokenizer_from_json


def tokenize(sentences) -> (np.ndarray, Tokenizer):
    # Create tokenizer
    text_tokenizer = Tokenizer()
    # Fit texts
    text_tokenizer.fit_on_texts(sentences)
    return text_tokenizer.texts_to_sequences(sentences), text_tokenizer


def clean_sentence(sentence):
    # Lower case the sentence
    lower_case_sent = sentence.lower()
    # Strip punctuation
    string_punctuation = string.punctuation + "¡" + '¿'
    cleaned_sentence = lower_case_sent.translate(str.maketrans('', '', string_punctuation))

    return cleaned_sentence


def load_tokenizer(tokenizer_file_path) -> Tokenizer:
    # Read the JSON from the file
    with open(tokenizer_file_path, "r", encoding="utf-8") as f:
        tokenizer_json = json.load(f)

    # Recreate the tokenizer
    tokenizer = tokenizer_from_json(tokenizer_json)

    return tokenizer


def save_tokenizer(tokenizer, path):
    tokenizer_json = tokenizer.to_json()
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))


def logits_to_sentence(logits, tokenizer):
    index_to_words = {idx: word for word, idx in tokenizer.word_index.items()}
    index_to_words[0] = '<empty>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])
