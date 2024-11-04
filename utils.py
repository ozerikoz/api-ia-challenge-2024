import re
import unicodedata
from nltk.corpus import stopwords
import nltk
import os

# Configurar o caminho para os dados do NLTK
nltk_data_path = '/opt/render/nltk_data'
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)

# Baixar stopwords apenas se necessário
try:
    stop_words = set(stopwords.words('portuguese'))
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)
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
