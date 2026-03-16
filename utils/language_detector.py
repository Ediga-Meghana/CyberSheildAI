from langdetect import detect, DetectorFactory

# Make detection deterministic
DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi',
    'te': 'Telugu',
    'es': 'Spanish',
    'ar': 'Arabic',
    'fr': 'French',
    'de': 'German',
}


def detect_language(text):
    """Detect the language of a text string."""
    try:
        lang_code = detect(text)
        lang_name = SUPPORTED_LANGUAGES.get(lang_code, 'Unknown')
        return lang_code, lang_name
    except Exception:
        return 'en', 'English'


def is_english(text):
    """Check if text is English."""
    code, _ = detect_language(text)
    return code == 'en'
