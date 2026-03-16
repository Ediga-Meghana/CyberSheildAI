import re


def tokenize_text(text):
    """Tokenize text into words using regex — no NLTK download needed."""
    if not isinstance(text, str):
        return []
    return re.findall(r'\b\w+\b', text.lower())


def tokenize_batch(texts):
    """Tokenize a list of texts."""
    return [tokenize_text(t) for t in texts]
