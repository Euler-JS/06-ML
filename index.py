from flask import Flask, render_template_string, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Carregar modelos
modelo = joblib.load('modelo_titanic.pkl')
scaler = joblib.load('scaler_titanic.pkl')
model_info = joblib.load('model_info.pkl')

# Template HTML
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="pt">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predição Titanic - ML</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .model-info {
            background: #f8f9fa;
            padding: 20px;
            border-bottom: 2px solid #e9ecef;
        }
        
        .model-info h3 {
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        
        .stat {
            background: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .stat-label {
            color: #6c757d;
            font-size: 0.9em;
            margin-bottom: 5px;
        }
        
        .stat-value {
            color: #667eea;
            font-size: 1.5em;
            font-weight: bold;
        }
        
        .form-section {
            padding: 30px;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 600;
        }
        
        input, select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s;
        }
        
        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        .btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .result {
            margin-top: 30px;
            padding: 25px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            color: white;
            text-align: center;
            display: none;
        }
        
        .result.show {
            display: block;
            animation: slideIn 0.5s ease;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .result h2 {
            font-size: 2em;
            margin-bottom: 10px;
        }
        
        .result .price {
            font-size: 3em;
            font-weight: bold;
            margin: 15px 0;
        }
        
        .footer {
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚢 Titanic Fare Predictor</h1>
            <p>Predição de Tarifa usando Machine Learning</p>
        </div>
        
        <div class="model-info">
            <h3>📊 Informações do Modelo</h3>
            <div class="stats">
                <div class="stat">
                    <div class="stat-label">Modelo</div>
                    <div class="stat-value">{{ model_info.model_name }}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">R² Score</div>
                    <div class="stat-value">{{ "%.4f"|format(model_info.r2_score) }}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">RMSE</div>
                    <div class="stat-value">${{ "%.2f"|format(model_info.rmse) }}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">MAE</div>
                    <div class="stat-value">${{ "%.2f"|format(model_info.mae) }}</div>
                </div>
            </div>
        </div>
        
        <div class="form-section">
            <form id="predictionForm">
                <div class="form-row">
                    <div class="form-group">
                        <label for="pclass">Classe do Bilhete:</label>
                        <select name="pclass" id="pclass" required>
                            <option value="">Selecione...</option>
                            <option value="1">1ª Classe</option>
                            <option value="2">2ª Classe</option>
                            <option value="3">3ª Classe</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="sex">Sexo:</label>
                        <select name="sex" id="sex" required>
                            <option value="">Selecione...</option>
                            <option value="0">Feminino</option>
                            <option value="1">Masculino</option>
                        </select>
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="age">Idade:</label>
                        <input type="number" name="age" id="age" min="0" max="100" step="0.1" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="survived">Sobreviveu?</label>
                        <select name="survived" id="survived" required>
                            <option value="">Selecione...</option>
                            <option value="0">Não</option>
                            <option value="1">Sim</option>
                        </select>
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="sibsp">Irmãos/Cônjuges a bordo:</label>
                        <input type="number" name="sibsp" id="sibsp" min="0" max="10" value="0" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="parch">Pais/Filhos a bordo:</label>
                        <input type="number" name="parch" id="parch" min="0" max="10" value="0" required>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="embarked">Porto de Embarque:</label>
                    <select name="embarked" id="embarked" required>
                        <option value="">Selecione...</option>
                        <option value="0">Cherbourg</option>
                        <option value="1">Queenstown</option>
                        <option value="2">Southampton</option>
                    </select>
                </div>
                
                <button type="submit" class="btn">🔮 Prever Tarifa</button>
            </form>
            
            <div class="result" id="result">
                <h2>💰 Tarifa Prevista</h2>
                <div class="price" id="predictedPrice">$0.00</div>
                <p>Esta é a estimativa da tarifa baseada nas características informadas</p>
            </div>
        </div>
        
        <div class="footer">
            <p>🤖 Powered by Machine Learning | Modelo: {{ model_info.model_name }}</p>
        </div>
    </div>
    
    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    document.getElementById('predictedPrice').textContent = 
                        '$' + result.predicted_fare.toFixed(2);
                    document.getElementById('result').classList.add('show');
                } else {
                    alert('Erro: ' + result.error);
                }
            } catch (error) {
                alert('Erro ao fazer predição: ' + error);
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extrair valores
        pclass = int(data['pclass'])
        sex = int(data['sex'])
        age = float(data['age'])
        sibsp = int(data['sibsp'])
        parch = int(data['parch'])
        survived = int(data['survived'])
        embarked = int(data['embarked'])
        
        # Calcular features derivadas
        family_size = sibsp + parch + 1
        is_alone = 1 if family_size == 1 else 0
        
        # Criar array com as features na ordem correta
        features = np.array([[pclass, sex, age, sibsp, parch, survived, family_size, is_alone, embarked]])
        
        # Normalizar
        features_scaled = scaler.transform(features)
        
        # Fazer predição
        prediction = modelo.predict(features_scaled)[0]
        
        return jsonify({
            'success': True,
            'predicted_fare': float(prediction)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/model-info')
def get_model_info():
    return jsonify(model_info)

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Iniciando servidor Flask...")
    print("=" * 60)
    print(f"📊 Modelo carregado: {model_info['model_name']}")
    print(f"🎯 R² Score: {model_info['r2_score']:.4f}")
    print(f"📈 RMSE: ${model_info['rmse']:.2f}")
    print("=" * 60)
    print("🌐 Acesse: http://localhost:5111")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5111)