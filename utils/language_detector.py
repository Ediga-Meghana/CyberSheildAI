from langdetect import detect, DetectorFactory

# Make detection deterministic
DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi',
    'te': 'Telugu',
    'ta': 'Tamil',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'bn': 'Bengali',
    'mr': 'Marathi',
    'ur': 'Urdu',
    'es': 'Spanish',
}


def detect_language(text):
    """Detect the language of a text string, including transliterated Roman scripts."""
    text_lower = text.lower()
    
    # Heuristic fallback for transliterated Hindi (Roman/Hinglish)
    hindi_roman = ['tum', 'kaise', 'ho', 'bewakoof', 'kya', 'hai', 'pagal', 'nahi', 'karo', 'bhai', 'madarchod', 'bhonsdike']
    if any(word in text_lower for word in hindi_roman):
        return 'hi', 'Hindi'
        
    # Heuristic fallback for transliterated Telugu (Roman/Tenglish)
    telugu_roman = ['nuvvu', 'nachaledhu', 'emi', 'ra', 'babu', 'cheppu', 'ledu', 'kadha', 'lampa', 'kothi', 'picha', 'kukkala']
    if any(word in text_lower for word in telugu_roman):
        return 'te', 'Telugu'

    try:
        lang_code = detect(text)
        if lang_code in SUPPORTED_LANGUAGES:
            return lang_code, SUPPORTED_LANGUAGES[lang_code]
        else:
            # Fallback to English if detect() returns something weird, never 'Unknown'
            return 'en', 'English'
    except Exception:
        return 'en', 'English'


def is_english(text):
    """Check if text is English."""
    code, _ = detect_language(text)
    return code == 'en'
