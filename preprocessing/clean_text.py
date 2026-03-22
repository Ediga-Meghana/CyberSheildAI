import re
import unicodedata
import emoji

# Built-in English stop words — no NLTK download needed
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
    'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
    'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'through', 'during', 'before', 'after', 'above',
    'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
    't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're',
    've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
    'won', 'wouldn'
}

# Simple suffix rules for basic lemmatization (no NLTK needed)
def simple_lemmatize(word):
    """Basic suffix-stripping lemmatizer."""
    if len(word) <= 3:
        return word
    if word.endswith('ies') and len(word) > 4:
        return word[:-3] + 'y'
    if word.endswith('es') and len(word) > 3:
        return word[:-2]
    if word.endswith('ing') and len(word) > 5:
        return word[:-3]
    if word.endswith('ed') and len(word) > 4:
        return word[:-2]
    if word.endswith('ly') and len(word) > 4:
        return word[:-2]
    if word.endswith('s') and not word.endswith('ss') and len(word) > 3:
        return word[:-1]
    return word


def remove_urls(text):
    return re.sub(r'http\S+|www\.\S+', '', text)


def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')


def remove_mentions_hashtags(text):
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    return text


def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z\s]', '', text)


def remove_stopwords(text):
    words = text.split()
    return ' '.join([w for w in words if w.lower() not in STOP_WORDS])


def lemmatize_text(text):
    words = text.split()
    return ' '.join([simple_lemmatize(w) for w in words])


def clean_text(text):
    """Full preprocessing pipeline — zero external downloads needed."""
    if not isinstance(text, str):
        return ''
    
    # Normalize unicode (very important for Hindi, Telugu, etc.)
    text = unicodedata.normalize('NFKC', text)
    
    text = text.lower()
    text = remove_urls(text)
    text = remove_emojis(text)
    text = remove_mentions_hashtags(text)
    text = remove_special_characters(text)
    text = remove_stopwords(text)
    # text = lemmatize_text(text) # Commenting out lemmatization as transformer models prefer intact words
    text = re.sub(r'\s+', ' ', text).strip()
    return text
