import re
import unicodedata
from nltk.corpus import stopwords

# Configurar stopwords em português
stop_words = set(stopwords.words('portuguese'))

def preprocess_text(text):
    # Converte para minúsculas
    text = text.lower()
    # Remove acentos
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    # Remove caracteres especiais
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove stopwords
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)
