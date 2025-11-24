# Importar bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Adicionados: import para carregar de forma robusta
import os
from pathlib import Path

# ==========================
# 1. CARREGAR DADOS
# ==========================
print("=" * 60)
print("ANÁLISE DO DATASET TITANIC - REGRESSÃO LINEAR")
print("=" * 60)

# Carregar o dataset (antes: pd.read_csv('TitanicDataset.csv'))
def load_dataset(candidates=None):
    base = Path(__file__).resolve().parent
    if candidates is None:
        candidates = [
            "Titanic-Dataset.csv",
            "TitanicDataset.csv",
            "Titanic-Dataset .csv",
            "titanic-dataset.csv",
            "titanicdataset.csv"
        ]
    # 1) procurar na mesma pasta do script
    for name in candidates:
        p = base / name
        if p.exists():
            return pd.read_csv(p)
    # 2) procurar no cwd
    for name in candidates:
        p = Path.cwd() / name
        if p.exists():
            return pd.read_csv(p)
    # 3) procurar qualquer CSV com 'titanic' no nome (script dir e cwd)
    for p in list(base.glob("*.csv")) + list(Path.cwd().glob("*.csv")):
        if "titanic" in p.name.lower():
            return pd.read_csv(p)
    # 4) falha com mensagem útil
    files_script = ", ".join([f.name for f in base.iterdir()])
    files_cwd = ", ".join([f.name for f in Path.cwd().iterdir()])
    raise FileNotFoundError(
        f"Dataset não encontrado. Procurei em:\n  script dir: {base}\n  cwd: {Path.cwd()}\n\n"
        f"Arquivos no script dir: {files_script}\nArquivos no cwd: {files_cwd}\n\n"
        "Coloque o arquivo 'Titanic-Dataset.csv' na pasta do projeto ou no diretório atual."
    )

df = load_dataset()

print("\n📊 Primeiras linhas do dataset:")
print(df.head())

print(f"\n📏 Dimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")

# ==========================
# 2. ANÁLISE EXPLORATÓRIA
# ==========================
print("\n" + "=" * 60)
print("ANÁLISE EXPLORATÓRIA DOS DADOS")
print("=" * 60)

print("\n🔍 Informações do dataset:")
print(df.info())

print("\n📈 Estatísticas descritivas:")
print(df.describe())

print("\n❓ Valores nulos:")
print(df.isnull().sum())

print("\n📊 Estatísticas da variável alvo (Fare):")
print(f"Média: ${df['Fare'].mean():.2f}")
print(f"Mediana: ${df['Fare'].median():.2f}")
print(f"Desvio padrão: ${df['Fare'].std():.2f}")
print(f"Mínimo: ${df['Fare'].min():.2f}")
print(f"Máximo: ${df['Fare'].max():.2f}")

print("\n📊 Tarifa média por Classe:")
print(df.groupby('Pclass')['Fare'].mean().sort_values(ascending=False))

print("\n📊 Tarifa média por Sexo:")
print(df.groupby('Sex')['Fare'].mean())

# ==========================
# 3. PRÉ-PROCESSAMENTO
# ==========================
print("\n" + "=" * 60)
print("PRÉ-PROCESSAMENTO DOS DADOS")
print("=" * 60)

# Criar cópia para processamento
df_processed = df.copy()

# Remover linhas onde Fare é nulo (variável alvo)
print(f"\n🔧 Removendo linhas com Fare nulo...")
print(f"Linhas antes: {len(df_processed)}")
df_processed = df_processed[df_processed['Fare'].notna()]
print(f"Linhas depois: {len(df_processed)}")

# Preencher valores nulos nas features
print("\n🔧 Tratando valores nulos nas features...")
df_processed['Age'].fillna(df_processed['Age'].median(), inplace=True)
df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0], inplace=True)

print("Valores nulos após tratamento:")
print(df_processed.isnull().sum())

# Remover coluna Cabin (muitos nulos)
if 'Cabin' in df_processed.columns:
    df_processed.drop('Cabin', axis=1, inplace=True)

# Codificar variáveis categóricas
print("\n🔧 Codificando variáveis categóricas...")
le_sex = LabelEncoder()
df_processed['Sex'] = le_sex.fit_transform(df_processed['Sex'])

if 'Embarked' in df_processed.columns:
    le_embarked = LabelEncoder()
    df_processed['Embarked'] = le_embarked.fit_transform(df_processed['Embarked'])

# Criar novas features
print("🔧 Criando novas features...")
df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
df_processed['IsAlone'] = (df_processed['FamilySize'] == 1).astype(int)

print(f"FamilySize - Min: {df_processed['FamilySize'].min()}, Max: {df_processed['FamilySize'].max()}")
print(f"IsAlone - Distribuição:\n{df_processed['IsAlone'].value_counts()}")

# Selecionar features para o modelo
features_to_use = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Survived', 'FamilySize', 'IsAlone']
if 'Embarked' in df_processed.columns:
    features_to_use.append('Embarked')

X = df_processed[features_to_use]
y = df_processed['Fare']

print(f"\n✅ Features selecionadas: {features_to_use}")
print(f"✅ Variável alvo: Fare (Tarifa)")
print(f"✅ Shape dos dados: X={X.shape}, y={y.shape}")

# ==========================
# 4. DIVISÃO TREINO/TESTE
# ==========================
print("\n" + "=" * 60)
print("DIVISÃO DOS DADOS")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📊 Tamanho do conjunto de treino: {X_train.shape[0]} amostras")
print(f"📊 Tamanho do conjunto de teste: {X_test.shape[0]} amostras")
print(f"\nEstatísticas da tarifa no treino:")
print(f"  Média: ${y_train.mean():.2f}")
print(f"  Mediana: ${y_train.median():.2f}")
print(f"\nEstatísticas da tarifa no teste:")
print(f"  Média: ${y_test.mean():.2f}")
print(f"  Mediana: ${y_test.median():.2f}")

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✅ Dados normalizados com StandardScaler")

# ==========================
# 5. TREINAMENTO DOS MODELOS
# ==========================
print("\n" + "=" * 60)
print("TREINAMENTO DOS MODELOS DE REGRESSÃO")
print("=" * 60)

# Dicionário de modelos
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0, random_state=42),
    'Lasso Regression': Lasso(alpha=1.0, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n🚀 Treinando {name}...")
    
    # Treinar modelo
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Cross-validation (R² score negativo no sklearn)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    
    # Calcular métricas
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'predictions': y_pred,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"✅ MSE (Mean Squared Error): {mse:.4f}")
    print(f"✅ RMSE (Root Mean Squared Error): ${rmse:.2f}")
    print(f"✅ MAE (Mean Absolute Error): ${mae:.2f}")
    print(f"✅ R² Score: {r2:.4f}")
    print(f"📊 Cross-validation R² (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ==========================
# 6. AVALIAÇÃO DOS MODELOS
# ==========================
print("\n" + "=" * 60)
print("AVALIAÇÃO E COMPARAÇÃO DOS MODELOS")
print("=" * 60)

# Comparar modelos
print("\n📊 Resumo dos Resultados:")
print("-" * 80)
print(f"{'Modelo':<20} {'RMSE':<12} {'MAE':<12} {'R²':<12} {'CV R²':<15}")
print("-" * 80)
for name, result in sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True):
    print(f"{name:<20} ${result['rmse']:<11.2f} ${result['mae']:<11.2f} {result['r2']:<12.4f} {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})")

# Melhor modelo
best_model_name = max(results, key=lambda x: results[x]['r2'])
best_model = results[best_model_name]['model']
best_predictions = results[best_model_name]['predictions']

print(f"\n🏆 Melhor modelo: {best_model_name}")
print(f"🎯 R² Score: {results[best_model_name]['r2']:.4f}")
print(f"🎯 RMSE: ${results[best_model_name]['rmse']:.2f}")
print(f"🎯 MAE: ${results[best_model_name]['mae']:.2f}")

# Análise de resíduos
residuals = y_test - best_predictions
print(f"\n📊 Análise de Resíduos ({best_model_name}):")
print("-" * 60)
print(f"Média dos resíduos: ${residuals.mean():.2f}")
print(f"Desvio padrão dos resíduos: ${residuals.std():.2f}")
print(f"Resíduo mínimo: ${residuals.min():.2f}")
print(f"Resíduo máximo: ${residuals.max():.2f}")

# Coeficientes do modelo
print(f"\n📊 Coeficientes do Modelo ({best_model_name}):")
print("-" * 60)
coef_df = pd.DataFrame({
    'Feature': features_to_use,
    'Coeficiente': best_model.coef_
}).sort_values('Coeficiente', key=abs, ascending=False)
print(coef_df.to_string(index=False))
print(f"\nIntercepto: ${best_model.intercept_:.2f}")

# ==========================
# 7. PREDIÇÕES DE EXEMPLO
# ==========================
print("\n" + "=" * 60)
print("EXEMPLOS DE PREDIÇÃO")
print("=" * 60)

# Mostrar alguns exemplos de predição
examples = X_test.head(10)
examples_scaled = scaler.transform(examples)
predictions = best_model.predict(examples_scaled)

print("\nPrimeiros 10 passageiros do conjunto de teste:")
print("-" * 80)
print(f"{'Real':<12} {'Predito':<12} {'Erro':<12} {'Classe':<8} {'Sexo':<8} {'Idade':<8}")
print("-" * 80)
for i in range(len(examples)):
    real = y_test.iloc[i]
    pred = predictions[i]
    erro = abs(real - pred)
    print(f"${real:<11.2f} ${pred:<11.2f} ${erro:<11.2f} {examples.iloc[i]['Pclass']:<8} {examples.iloc[i]['Sex']:<8} {examples.iloc[i]['Age']:<8.0f}")

# ==========================
# 8. EXPORTAR O MODELO
# ==========================
print("\n" + "=" * 60)
print("EXPORTAR MODELO E SCALER")
print("=" * 60)

import joblib

# Exportar o melhor modelo
joblib.dump(best_model, 'modelo_titanic.pkl')
print(f"✅ Modelo exportado: modelo_titanic.pkl")

# Exportar o scaler
joblib.dump(scaler, 'scaler_titanic.pkl')
print(f"✅ Scaler exportado: scaler_titanic.pkl")

# Exportar informações adicionais
model_info = {
    'model_name': best_model_name,
    'features': features_to_use,
    'r2_score': results[best_model_name]['r2'],
    'rmse': results[best_model_name]['rmse'],
    'mae': results[best_model_name]['mae']
}
joblib.dump(model_info, 'model_info.pkl')
print(f"✅ Informações do modelo exportadas: model_info.pkl")

print("\n💡 Para carregar o modelo depois:")
print("   modelo = joblib.load('modelo_titanic.pkl')")
print("   scaler = joblib.load('scaler_titanic.pkl')")
print("   info = joblib.load('model_info.pkl')")

# ==========================
# 9. CONCLUSÕES
# ==========================
print("\n" + "=" * 60)
print("CONCLUSÕES")
print("=" * 60)
print(f"""
✅ Modelo de regressão treinado com sucesso!
🎯 Melhor modelo: {best_model_name}
📊 R² Score: {results[best_model_name]['r2']:.4f}
📊 RMSE: ${results[best_model_name]['rmse']:.2f}
📊 MAE: ${results[best_model_name]['mae']:.2f}
🔍 Total de passageiros analisados: {len(df_processed)}
📈 Cross-validation R²: {results[best_model_name]['cv_mean']:.4f}

💡 O modelo pode prever a tarifa (Fare) paga por um passageiro
   com base em características como classe, sexo, idade, família
   e status de sobrevivência.
   
📌 Interpretação do R²: {results[best_model_name]['r2']*100:.2f}% da variabilidade
   da tarifa é explicada pelo modelo.
""")

print("=" * 60)
print("ANÁLISE CONCLUÍDA!")
print("=" * 60)