from flask import Flask, request, jsonify
import pandas as pd
import pickle
from model import diagnostic
from utils import preprocess_text

app = Flask(__name__)

# Carregar o modelo
model_filename = 'word2vec_model.pkl'

with open(model_filename, 'rb') as file:
    model = pickle.load(file)
    
# Carregar o DataFrame do CSV
csv_path = 'dataset.csv' 
df = pd.read_csv(csv_path, sep=",")

train_problems_processed = df['Problema'].apply(lambda x: preprocess_text(x)).tolist()  
train_df = df.copy()

@app.route('/')
def home():
    return "Bem-vindo ao modelo de diagnostico de problemas automotivos!"

@app.route('/diagnostic', methods=['POST'])
def diagnostic_route():
    data = request.json
    problema = data.get('problema')
    modelo_usuario = data.get('modelo')
    fabricante_usuario = data.get('fabricante')
    ano_usuario = data.get('ano')

    if not problema:
        return jsonify({'error': 'Problema não fornecido.'}), 400
    
    process_problema = preprocess_text(problema)

    result = diagnostic(process_problema, train_problems_processed, train_df, model, modelo_usuario, fabricante_usuario, ano_usuario)

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
        
        return jsonify({'resultado': result}), 200  
    else:
        return jsonify({'mensagem': 'Nenhum resultado encontrado.'}), 404

if __name__ == '__main__':
    app.run(debug=True)
    
    
    
