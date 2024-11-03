import re
import unicodedata
from sentence_transformers import util
import nltk
from nltk.corpus import stopwords

# Baixar stopwords em português se necessário
nltk.download('stopwords')
stop_words = set(stopwords.words('portuguese'))

# Função de pré-processamento para limpar e normalizar o texto
def preprocess_text(text):
    # Converte para minúsculas
    text = text.lower()
    
    # Remove acentos
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    
    # Remove caracteres especiais
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Remove stopwords
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Função para identificar a categoria do problema com base no texto do usuário
def identify_category(user_text, categories):
    normalized_categories = [preprocess_text(cat) for cat in categories]  # Pré-processa as categorias
    user_text = preprocess_text(user_text)  # Pré-processa o texto do usuário
    for category in normalized_categories:
        if category in user_text:  # Verifica se a categoria está presente no texto do usuário
            return category
    return None  # Retorna None se nenhuma categoria for identificada

# Função para encontrar o problema mais próximo com base na similaridade
def find_similar_problem(user_input, df, model, similarity_threshold=0.7, fabricante=None, modelo=None, ano=None):
    # Identificar a categoria do problema do usuário
    df['Categoria do Problema'] = df['Categoria do Problema'].apply(preprocess_text)
    categories = df['Categoria do Problema'].unique()
    category = identify_category(user_input, categories)

    if category:
        # Filtrar o conjunto de dados para a categoria identificada
        category_df = df[df['Categoria do Problema'] == category].copy()
    else:
        # Se nenhuma categoria for encontrada, considerar todo o dataset
        category_df = df.copy()

    # Pré-processar os problemas da categoria filtrada
    category_df['Problema_Processed'] = category_df['Problema'].apply(preprocess_text)
    
    # Criar embeddings para os problemas da categoria filtrada
    category_df['Problema_Embedding'] = category_df['Problema_Processed'].apply(lambda x: model.encode(x, convert_to_tensor=True))

    # Criar o embedding para o problema do usuário
    user_embedding = model.encode(preprocess_text(user_input), convert_to_tensor=True)

    # Calcular similaridade entre o problema do usuário e os problemas da categoria
    similarities = []
    for _, row in category_df.iterrows():
        similarity = util.pytorch_cos_sim(user_embedding, row['Problema_Embedding']).item()
        similarities.append((row, similarity))
    
   # Ordena por similaridade em ordem decrescente
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    
    # Filtrar problemas com similaridade acima do limite
    high_similarity_problems = [(row, similarity) for row, similarity in sorted_similarities if similarity >= similarity_threshold]
    
    if high_similarity_problems:
        # Se múltiplos problemas similares forem encontrados, aplicar filtro adicional por Fabricante, Modelo e Ano
        if fabricante or modelo or ano:
            filtered_problems = [
                (row, similarity) for row, similarity in high_similarity_problems
                   if (modelo is None or row.get('Modelo') == modelo) and
                   (fabricante is None or row.get('Fabricante') == fabricante) and
                   (ano is None or row.get('Ano') == ano)
            ]
            # Se problemas foram filtrados por Fabricante, Modelo e Ano, use o mais similar entre eles
            if filtered_problems:
                best_match_row, best_similarity = max(filtered_problems, key=lambda x: x[1])
                return best_match_row, best_similarity
    
        # Caso não tenha conseguido filtrar, retorna o mais similar entre os problemas encontrados
        best_match_row, best_similarity = high_similarity_problems[0]
        return best_match_row, best_similarity
    else:
        return None, 0  # Se a similaridade for insuficiente
