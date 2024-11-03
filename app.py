from flask import Flask, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer
from modelo_ia.model import find_similar_problem  # Certifique-se de que o model.py está em api/

# Carregar o dataset e o modelo SBERT
df = pd.read_csv('modelo_ia\dataset_ia.csv', sep=";")  # Ajuste o caminho se necessário
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Inicializar a aplicação Flask
app = Flask(__name__)

@app.route('/similarity', methods=['POST'])
def similarity():
    data = request.get_json()
    user_problem = data.get('user_problem')
    fabricante = data.get('fabricante')
    modelo = data.get('modelo')
    ano = data.get('ano')

    if not user_problem:
        return jsonify({'error': 'O campo user_problem é obrigatório.'}), 400

    # Encontrar o problema mais similar usando a função de similaridade
    best_match_row, similarity = find_similar_problem(user_problem, df, model, fabricante=fabricante, modelo=modelo, ano=ano)

    if best_match_row is not None:
        result = {
            "fabricante": best_match_row['Fabricante'],
            "modelo": best_match_row['Modelo'],
            "ano": int(best_match_row['Ano']),
            "problema": best_match_row['Problema'],
            "causa": best_match_row['Causa'],
            "solucao": best_match_row['Solucao'],
            "orcamento": float(best_match_row['Orcamento']),
            "categoria": best_match_row['Categoria do Problema'],
            "gravidade": best_match_row['Gravidade do Problema'],
            "tempo_reparo": best_match_row['Tempo Estimado de Reparo'],
            "similarity": round(similarity, 2)
        }
    else:
        result = {'message': 'Nenhum problema similar encontrado.'}

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)