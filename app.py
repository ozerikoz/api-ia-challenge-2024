from flask import Flask, request, jsonify
import pandas as pd
from modelo_ia.model import load_word2vec_model, get_average_vector, test_model_with_new_sample

# Criar a aplicação Flask
app = Flask(__name__)

# Carregar o modelo Word2Vec
model_path = 'modelo_ia/word2vec_model.pkl' 
model = load_word2vec_model(model_path)

# Carregar o DataFrame do CSV
csv_path = 'modelo_ia/dataset.csv'  # Substitua pelo caminho do seu CSV
df = pd.read_csv(csv_path, sep=",")
print(df.head())
train_problems_processed = df['Problema'].apply(lambda x: x.lower()).tolist()  # Pré-processamento simples
train_df = df.copy()

# Endpoint para receber a nova amostra
@app.route('/similarity', methods=['POST'])
def test():
    data = request.json
    new_sample = data.get('problema')
    modelo_usuario = data.get('modelo')
    fabricante_usuario = data.get('fabricante')
    ano_usuario = data.get('ano')

    if not new_sample:
        return jsonify({'error': 'Nova amostra não fornecida.'}), 400

    result = test_model_with_new_sample(new_sample, train_problems_processed, train_df, model, modelo_usuario, fabricante_usuario, ano_usuario)

    if result is not None:
        result = {
            "fabricante": result['Fabricante'],
            "modelo": result['Modelo'],
            "ano": int(result['Ano']),
            "problema": result['Problema'],
            "categoria": result['Categoria do Problema'],
            "gravidade": result['Gravidade do Problema'],
            "causa": result['Causa'],
            "orcamento": result['Orcamento'],
            "similaridade": round(result['Similaridade'], 2),
            "solucao": result['Solucao'],
            "tempo_reparo": result['Tempo Estimado de Reparo']
        }
        
        return jsonify({'resultado': result}), 200  # Retorna a linha inteira do dataset
    else:
        return jsonify({'mensagem': 'Nenhum resultado encontrado.'}), 404

# Inicializar a aplicação
if __name__ == '__main__':
    app.run(debug=True)
