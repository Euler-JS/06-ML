# 🫀 Sistema Completo de Predição de Risco Cardíaco

Sistema de Machine Learning para predição de risco de doença cardíaca baseado em indicadores clínicos, demográficos e hábitos de vida.

## 📋 Visão Geral

Este sistema completo inclui:
- ✅ Análise exploratória de dados
- ✅ Pré-processamento robusto
- ✅ Treinamento de múltiplos modelos de classificação
- ✅ Avaliação e comparação de modelos
- ✅ Interface gráfica interativa para predições
- ✅ Exportação de modelos treinados

## 🚀 Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Instalar Dependências

```bash
pip install pandas numpy scikit-learn joblib
```

Ou usando o ambiente virtual (recomendado):

```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
pip install pandas numpy scikit-learn joblib
```

## 📊 Dataset

O sistema utiliza o arquivo `heart_disease.csv` que contém 10.000+ registros de pacientes com as seguintes features:

### Características Demográficas
- **Age**: Idade do paciente (anos)
- **Gender**: Sexo (Male/Female)

### Indicadores Clínicos
- **Blood Pressure**: Pressão arterial (mmHg)
- **Cholesterol Level**: Nível de colesterol total (mg/dL)
- **BMI**: Índice de Massa Corporal (kg/m²)
- **High Blood Pressure**: Hipertensão (Yes/No)
- **Low HDL Cholesterol**: HDL baixo (Yes/No)
- **High LDL Cholesterol**: LDL alto (Yes/No)
- **Triglyceride Level**: Nível de triglicerídeos (mg/dL)
- **Fasting Blood Sugar**: Glicemia em jejum (mg/dL)
- **CRP Level**: Proteína C-Reativa (mg/L)
- **Homocysteine Level**: Nível de homocisteína (µmol/L)

### Hábitos de Vida
- **Exercise Habits**: Hábitos de exercício (Low/Medium/High)
- **Smoking**: Tabagismo (Yes/No)
- **Alcohol Consumption**: Consumo de álcool (None/Low/Medium/High)
- **Sleep Hours**: Horas de sono por dia
- **Sugar Consumption**: Consumo de açúcar (Low/Medium/High)
- **Stress Level**: Nível de estresse (Low/Medium/High)

### Histórico Médico
- **Family Heart Disease**: Histórico familiar (Yes/No)
- **Diabetes**: Diabetes (Yes/No)

### Variável Alvo
- **Heart Disease Status**: Status de doença cardíaca (Yes/No)

## 🔧 Uso

### 1. Treinar o Modelo

Primeiro, execute o script de análise e treinamento:

```bash
python heart_disease_analysis.py
```

Este script irá:
1. Carregar e analisar o dataset
2. Fazer pré-processamento dos dados
3. Treinar 4 modelos diferentes (Logistic Regression, Random Forest, Gradient Boosting, SVM)
4. Avaliar e comparar os modelos
5. Exportar o melhor modelo para a pasta `models/`

**Saída esperada:**
- Estatísticas descritivas do dataset
- Análise de valores nulos
- Métricas de desempenho de cada modelo
- Relatório de classificação detalhado
- Matriz de confusão
- Importância das features
- Modelos exportados em `models/`

### 2. Usar a Interface Gráfica

Após treinar o modelo, execute a interface gráfica:

```bash
python heart_disease_gui.py
```

A interface permite:
- ✅ Carregar automaticamente o modelo treinado
- ✅ Inserir dados de um paciente
- ✅ Fazer predição de risco cardíaco
- ✅ Ver probabilidades e confiança da predição
- ✅ Obter recomendações baseadas no resultado

## 📁 Estrutura de Arquivos

```
trabalho_final/
├── heart_disease.csv              # Dataset de treinamento
├── heart_disease_analysis.py      # Script de análise e treinamento
├── heart_disease_gui.py           # Interface gráfica
├── README.md                       # Este arquivo
└── models/                         # Modelos treinados (criado após execução)
    ├── heart_disease_model.pkl           # Melhor modelo
    ├── heart_disease_scaler.pkl          # Scaler para normalização
    ├── heart_disease_model_info.pkl      # Informações do modelo
    └── all_models_results.pkl            # Resultados de todos os modelos
```

## 🎯 Modelos Implementados

O sistema treina e compara 4 modelos de classificação:

1. **Logistic Regression**: Modelo linear simples e interpretável
2. **Random Forest**: Ensemble de árvores de decisão
3. **Gradient Boosting**: Boosting com otimização de gradiente
4. **SVM (Support Vector Machine)**: Classificador de margem máxima

**Métricas de Avaliação:**
- Acurácia (Accuracy)
- Precisão (Precision)
- Recall (Sensibilidade)
- F1-Score
- ROC AUC
- Cross-validation (5-fold)
- Matriz de confusão

## 💡 Exemplo de Uso da GUI

1. Execute `python heart_disease_gui.py`
2. A interface carrega automaticamente o modelo treinado
3. Preencha os campos com dados do paciente:
   - Idade: 56
   - Sexo: Male
   - Pressão Arterial: 153
   - Colesterol: 155
   - IMC: 24.99
   - etc.
4. Clique em "FAZER PREDIÇÃO"
5. Veja o resultado:
   - ⚠️ ALTO RISCO ou ✅ BAIXO RISCO
   - Probabilidades de cada classe
   - Confiança da predição
   - Recomendações médicas

## 📊 Interpretação dos Resultados

### Alto Risco (Prediction = 1)
- O paciente apresenta características associadas a maior risco cardiovascular
- **Recomendações:**
  - Consulta médica especializada urgente
  - Exames cardiológicos complementares
  - Avaliação de fatores de risco modificáveis
  - Possível intervenção preventiva ou terapêutica

### Baixo Risco (Prediction = 0)
- O paciente apresenta perfil de baixo risco cardiovascular
- **Recomendações:**
  - Manter hábitos saudáveis
  - Check-ups periódicos
  - Controle contínuo dos fatores de risco
  - Atividade física regular

## ⚠️ Avisos Importantes

1. **Este sistema é uma ferramenta de apoio à decisão clínica**
2. **Não substitui a avaliação de um profissional de saúde**
3. **O diagnóstico definitivo deve ser feito por um médico**
4. **As predições têm caráter orientativo e educacional**

## 🔬 Características Técnicas

### Pré-processamento
- Tratamento de valores nulos (mediana para numéricos, moda para categóricos)
- Codificação de variáveis categóricas (Label Encoding)
- Normalização com StandardScaler
- Balanceamento via stratified split

### Validação
- Split 80/20 (treino/teste)
- Stratified sampling para manter proporção de classes
- Cross-validation com 5 folds
- Métricas múltiplas para avaliação robusta

### Modelo Final
- Seleção automática do modelo com melhor acurácia
- Exportação de modelo, scaler e metadados
- Sistema de carregamento robusto na GUI

## 🐛 Solução de Problemas

### Erro: "Dataset não encontrado"
- Certifique-se de que `heart_disease.csv` está na pasta `trabalho_final/`
- O script procura automaticamente por nomes similares

### Erro: "Modelo não foi carregado"
- Execute primeiro `heart_disease_analysis.py` para treinar o modelo
- Verifique se a pasta `models/` foi criada com os arquivos .pkl

### Erro: "ModuleNotFoundError"
- Instale as dependências: `pip install pandas numpy scikit-learn joblib`

### Interface não abre
- Verifique se tkinter está instalado (geralmente vem com Python)
- No Ubuntu/Debian: `sudo apt-get install python3-tk`

## 📈 Melhorias Futuras

- [ ] Adicionar mais modelos (XGBoost, Neural Networks)
- [ ] Implementar feature engineering avançado
- [ ] Adicionar explicabilidade (SHAP, LIME)
- [ ] Criar API REST para integração
- [ ] Dashboard web com Flask/Streamlit
- [ ] Sistema de logging e monitoramento
- [ ] Testes automatizados
- [ ] Otimização de hiperparâmetros com GridSearch

## 📝 Licença

Este projeto é de código aberto e pode ser usado para fins educacionais e de pesquisa.

## 👥 Contribuições

Contribuições são bem-vindas! Sinta-se livre para:
- Reportar bugs
- Sugerir melhorias
- Adicionar novos modelos
- Melhorar a documentação

## 📞 Contato

Para dúvidas ou sugestões, entre em contato através do repositório.

---

**Desenvolvido com ❤️ para ajudar na prevenção de doenças cardíacas**
