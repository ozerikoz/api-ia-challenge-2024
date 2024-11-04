import requests
import json

def diagnostic(data):
    url = 'http://127.0.0.1:5000/diagnostic'
    headers = {'Content-type': 'application/json'}
    
    # Enviar a solicitação POST com os dados da nova amostra
    response = requests.post(url, data=json.dumps(data), headers=headers)
    
    # Verificar se a solicitação foi bem-sucedida
    if response.status_code == 200:
        result = response.json()

        diagnostic = result['resultado']
        json_formated = json.dumps(diagnostic, indent=4, ensure_ascii=False)
        
        print(f"\nResultado do diagnostico: {json_formated}\n")
    else:
        print("Erro ao fazer a previsão.")

if __name__ == '__main__':
    # Dados da nova amostra
    nova_amostra = {
        "problema": "motor esquentando",
        "fabricante": "Toyota",
        "modelo": "Corolla",
        "ano": 2018
    }

    # Realizar o diagnostico
    diagnostic(nova_amostra)
