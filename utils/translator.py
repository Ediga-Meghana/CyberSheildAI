from deep_translator import GoogleTranslator


def translate_to_english(text, source_lang='auto'):
    """Translate text to English using Google Translate."""
    try:
        if source_lang == 'en':
            return text
        translator = GoogleTranslator(source=source_lang, target='en')
        translated = translator.translate(text)
        return translated if translated else text
    except Exception:
        return text


def translate_text(text, source='auto', target='en'):
    """General translation function."""
    try:
        translator = GoogleTranslator(source=source, target=target)
        return translator.translate(text)
    except Exception:
        return text
