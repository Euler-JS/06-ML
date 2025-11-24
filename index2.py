from flask import Flask, render_template_string, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Carregar modelos
modelo = joblib.load('modelo_titanic_survival.pkl')
scaler = joblib.load('scaler_titanic_survival.pkl')
model_info = joblib.load('model_info_survival.pkl')

# Template HTML
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="pt">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predição Sobrevivência Titanic</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.8em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.95;
        }
        
        .model-info {
            background: #f8f9fa;
            padding: 25px;
            border-bottom: 3px solid #e9ecef;
        }
        
        .model-info h3 {
            color: #1e3c72;
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        
        .stat {
            background: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 3px 15px rgba(0,0,0,0.1);
            border-left: 4px solid #1e3c72;
        }
        
        .stat-label {
            color: #6c757d;
            font-size: 0.95em;
            margin-bottom: 8px;
            font-weight: 600;
        }
        
        .stat-value {
            color: #1e3c72;
            font-size: 1.8em;
            font-weight: bold;
        }
        
        .form-section {
            padding: 35px;
        }
        
        .form-group {
            margin-bottom: 25px;
        }
        
        label {
            display: block;
            margin-bottom: 10px;
            color: #333;
            font-weight: 600;
            font-size: 1.05em;
        }
        
        input, select {
            width: 100%;
            padding: 14px;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            font-size: 1em;
            transition: all 0.3s;
        }
        
        input:focus, select:focus {
            outline: none;
            border-color: #1e3c72;
            box-shadow: 0 0 0 3px rgba(30, 60, 114, 0.1);
        }
        
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
        }
        
        .btn {
            width: 100%;
            padding: 18px;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 1.2em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            margin-top: 10px;
        }
        
        .btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(30, 60, 114, 0.4);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .result {
            margin-top: 35px;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            display: none;
        }
        
        .result.show {
            display: block;
            animation: slideIn 0.5s ease;
        }
        
        .result.survived {
            background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
            color: white;
        }
        
        .result.not-survived {
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
            color: white;
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
            font-size: 2.5em;
            margin-bottom: 15px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .result .probability {
            font-size: 4em;
            font-weight: bold;
            margin: 20px 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .result .message {
            font-size: 1.3em;
            margin-top: 15px;
            opacity: 0.95;
        }
        
        .footer {
            background: #f8f9fa;
            padding: 25px;
            text-align: center;
            color: #6c757d;
            font-size: 0.95em;
        }
        
        .icon {
            font-size: 3em;
            margin-bottom: 15px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚢 Titanic Survival Predictor</h1>
            <p>Descubra suas chances de sobrevivência no Titanic</p>
        </div>
        
        <div class="model-info">
            <h3>📊 Métricas do Modelo de Machine Learning</h3>
            <div class="stats">
                <div class="stat">
                    <div class="stat-label">Modelo</div>
                    <div class="stat-value" style="font-size: 1.2em;">{{ model_info.model_name }}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Acurácia</div>
                    <div class="stat-value">{{ "%.1f"|format(model_info.accuracy * 100) }}%</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Precisão</div>
                    <div class="stat-value">{{ "%.1f"|format(model_info.precision * 100) }}%</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Recall</div>
                    <div class="stat-value">{{ "%.1f"|format(model_info.recall * 100) }}%</div>
                </div>
                <div class="stat">
                    <div class="stat-label">F1-Score</div>
                    <div class="stat-value">{{ "%.3f"|format(model_info.f1_score) }}</div>
                </div>
            </div>
        </div>
        
        <div class="form-section">
            <form id="predictionForm">
                <div class="form-row">
                    <div class="form-group">
                        <label for="pclass">🎫 Classe do Bilhete:</label>
                        <select name="pclass" id="pclass" required>
                            <option value="">Selecione...</option>
                            <option value="1">1ª Classe (Luxo)</option>
                            <option value="2">2ª Classe (Média)</option>
                            <option value="3">3ª Classe (Econômica)</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="sex">👤 Sexo:</label>
                        <select name="sex" id="sex" required>
                            <option value="">Selecione...</option>
                            <option value="0">Feminino</option>
                            <option value="1">Masculino</option>
                        </select>
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="age">🎂 Idade:</label>
                        <input type="number" name="age" id="age" min="0" max="100" step="0.5" placeholder="Ex: 25" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="fare">💰 Tarifa (£):</label>
                        <input type="number" name="fare" id="fare" min="0" step="0.01" placeholder="Ex: 50.00" required>
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="sibsp">👨‍👩‍👧 Irmãos/Cônjuges a bordo:</label>
                        <input type="number" name="sibsp" id="sibsp" min="0" max="10" value="0" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="parch">👨‍👩‍👧‍👦 Pais/Filhos a bordo:</label>
                        <input type="number" name="parch" id="parch" min="0" max="10" value="0" required>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="embarked">⚓ Porto de Embarque:</label>
                    <select name="embarked" id="embarked" required>
                        <option value="">Selecione...</option>
                        <option value="0">Cherbourg (França)</option>
                        <option value="1">Queenstown (Irlanda)</option>
                        <option value="2">Southampton (Inglaterra)</option>
                    </select>
                </div>
                
                <button type="submit" class="btn">🔮 Prever Sobrevivência</button>
            </form>
            
            <div class="result" id="result">
                <div class="icon" id="resultIcon">⚓</div>
                <h2 id="resultTitle">Resultado</h2>
                <div class="probability" id="probability">0%</div>
                <div class="message" id="message">Mensagem</div>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>🤖 Powered by Machine Learning</strong></p>
            <p>Modelo: {{ model_info.model_name }} | Acurácia: {{ "%.1f"|format(model_info.accuracy * 100) }}%</p>
            <p style="margin-top: 10px; font-size: 0.9em;">
                ⚠️ Esta é uma predição baseada em dados históricos do Titanic para fins educacionais
            </p>
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
                    const survived = result.survived;
                    const probability = result.probability;
                    
                    const resultDiv = document.getElementById('result');
                    const icon = document.getElementById('resultIcon');
                    const title = document.getElementById('resultTitle');
                    const probDiv = document.getElementById('probability');
                    const message = document.getElementById('message');
                    
                    // Remover classes anteriores
                    resultDiv.classList.remove('survived', 'not-survived');
                    
                    if (survived) {
                        resultDiv.classList.add('survived');
                        icon.textContent = '✅';
                        title.textContent = 'VOCÊ SOBREVIVERIA!';
                        message.textContent = 'Parabéns! As chances estão a seu favor.';
                    } else {
                        resultDiv.classList.add('not-survived');
                        icon.textContent = '❌';
                        title.textContent = 'VOCÊ NÃO SOBREVIVERIA';
                        message.textContent = 'Infelizmente, as chances não estão a seu favor.';
                    }
                    
                    probDiv.textContent = probability.toFixed(1) + '%';
                    resultDiv.classList.add('show');
                    
                    // Scroll suave para o resultado
                    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
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
        fare = float(data['fare'])
        embarked = int(data['embarked'])
        
        # Calcular features derivadas
        family_size = sibsp + parch + 1
        is_alone = 1 if family_size == 1 else 0
        
        # Criar array com as features na ordem correta
        # ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone', 'Embarked']
        features = np.array([[pclass, sex, age, sibsp, parch, fare, family_size, is_alone, embarked]])
        
        # Normalizar
        features_scaled = scaler.transform(features)
        
        # Fazer predição
        prediction = modelo.predict(features_scaled)[0]
        probability = modelo.predict_proba(features_scaled)[0]
        
        # Probabilidade de sobrevivência (classe 1)
        survival_probability = probability[1] * 100
        
        return jsonify({
            'success': True,
            'survived': bool(prediction),
            'probability': float(survival_probability),
            'prediction_value': int(prediction)
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
    print("🚀 Iniciando servidor Flask - Predição de Sobrevivência")
    print("=" * 60)
    print(f"📊 Modelo carregado: {model_info['model_name']}")
    print(f"🎯 Acurácia: {model_info['accuracy']*100:.2f}%")
    print(f"🎯 Precisão: {model_info['precision']*100:.2f}%")
    print(f"🎯 Recall: {model_info['recall']*100:.2f}%")
    print(f"🎯 F1-Score: {model_info['f1_score']:.4f}")
    print("=" * 60)
    print("🌐 Acesse: http://localhost:5112")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5112)