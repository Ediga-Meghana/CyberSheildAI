import re
import unicodedata

def clean_text(text: str) -> str:
    """
    Clean text for multilingual transformer models.
    Removes URLs, user mentions, and extra spaces.
    Preserves Unicode (Telugu, Hindi, emojis, etc.) so the multilingual model can understand it.
    """
    if not isinstance(text, str):
        return ""
        
    # Normalize unicode to ensure consistent character representation
    text = unicodedata.normalize('NFKC', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # Note: Emojis and special characters are preserved!
    # Note: Case is preserved!
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
