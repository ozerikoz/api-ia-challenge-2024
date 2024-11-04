
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Função para calcular o vetor médio de uma sentença
def get_average_vector(model, processed_text):
    words = processed_text.split()
    word_vectors = []
    
    for word in words:
        if word in model.wv.key_to_index:
            word_vectors.append(model.wv[word])

    if not word_vectors:
        return np.zeros(model.vector_size)

    average_vector = np.mean(word_vectors, axis=0)
    return average_vector

# Função de teste
def diagnostic(new_sample, train_problems_processed, train_df, model, modelo_usuario=None, fabricante_usuario=None, ano_usuario=None):
    processed_sample = new_sample.lower()  # Pré-processar
    new_sample_vector = get_average_vector(model, processed_sample)

    train_vectors = [get_average_vector(model, problem) for problem in train_problems_processed]
    similarities = cosine_similarity([new_sample_vector], train_vectors)[0]

    max_similarity_idx = np.argmax(similarities)
    most_similar_problem = train_df.iloc[max_similarity_idx].copy()  # Captura a linha inteira

    # Adiciona a similaridade à linha
    most_similar_problem['Similaridade'] = similarities[max_similarity_idx]


    # Verificar se a similaridade é menor que 0.5
    if similarities[max_similarity_idx] < 0.5:
        print("Nenhum resultado encontrado com similaridade abaixo de 0.5.")
        return None  # Retorna None se a similaridade for menor que 0.5

    # Aplicar filtro
    if modelo_usuario and most_similar_problem['Modelo'] == modelo_usuario:
        if ano_usuario and most_similar_problem['Ano'] == ano_usuario:
            return most_similar_problem.to_dict()  # Retorna a linha filtrada como dicionário
    elif fabricante_usuario and most_similar_problem['Fabricante'] == fabricante_usuario:
        if ano_usuario and most_similar_problem['Ano'] == ano_usuario:
            return most_similar_problem.to_dict()  # Retorna a linha filtrada como dicionário

    return most_similar_problem.to_dict()  # Retorna a linha inteira do dataset se nenhuma condição for atendida

    return None  # Caso não encontre correspondências
