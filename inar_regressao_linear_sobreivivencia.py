# Importar bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')
import os
from pathlib import Path

# ==========================
# 1. CARREGAR DADOS
# ==========================
print("=" * 60)
print("ANÁLISE DO DATASET TITANIC - PREDIÇÃO DE SOBREVIVÊNCIA")
print("=" * 60)

# Carregar o dataset de forma robusta
def load_dataset(candidates=None):
    base = Path(__file__).resolve().parent
    if candidates is None:
        candidates = [
            'Titanic-Dataset.csv',
            'TitanicDataset.csv',
            'Titanic-Dataset .csv',
            'titanic-dataset.csv',
            'titanicdataset.csv',
            'Titanic-Dataset.CSV',
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
    for p in list(base.glob('*.csv')) + list(Path.cwd().glob('*.csv')):
        if 'titanic' in p.name.lower():
            return pd.read_csv(p)

    # 4) falha com mensagem útil
    files_script = ', '.join([f.name for f in base.iterdir()])
    files_cwd = ', '.join([f.name for f in Path.cwd().iterdir()])
    raise FileNotFoundError(
        f"Dataset não encontrado. Procurei em:\n  script dir: {base}\n  cwd: {Path.cwd()}\n\n"
        f"Arquivos no script dir: {files_script}\nArquivos no cwd: {files_cwd}\n\n"
        "Coloque o arquivo 'Titanic-Dataset.csv' na pasta do projeto ou no diretório atual."
    )


# Carregar o DataFrame
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

print("\n📊 Distribuição da variável alvo (Survived):")
print(df['Survived'].value_counts())
print(f"\nTaxa de sobrevivência: {df['Survived'].mean()*100:.2f}%")

print("\n📊 Taxa de sobrevivência por Classe:")
print(df.groupby('Pclass')['Survived'].mean().sort_values(ascending=False))

print("\n📊 Taxa de sobrevivência por Sexo:")
print(df.groupby('Sex')['Survived'].mean())

# ==========================
# 3. PRÉ-PROCESSAMENTO
# ==========================
print("\n" + "=" * 60)
print("PRÉ-PROCESSAMENTO DOS DADOS")
print("=" * 60)

# Criar cópia para processamento
df_processed = df.copy()

# Remover linhas onde Survived é nulo (variável alvo)
print(f"\n🔧 Removendo linhas com Survived nulo...")
print(f"Linhas antes: {len(df_processed)}")
df_processed = df_processed[df_processed['Survived'].notna()]
print(f"Linhas depois: {len(df_processed)}")

# Preencher valores nulos nas features
print("\n🔧 Tratando valores nulos nas features...")
df_processed['Age'].fillna(df_processed['Age'].median(), inplace=True)
df_processed['Fare'].fillna(df_processed['Fare'].median(), inplace=True)
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

# Selecionar features para o modelo (SEM a variável Survived)
features_to_use = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone']
if 'Embarked' in df_processed.columns:
    features_to_use.append('Embarked')

X = df_processed[features_to_use]
y = df_processed['Survived']

print(f"\n✅ Features selecionadas: {features_to_use}")
print(f"✅ Variável alvo: Survived (Sobrevivência)")
print(f"✅ Shape dos dados: X={X.shape}, y={y.shape}")

# ==========================
# 4. DIVISÃO TREINO/TESTE
# ==========================
print("\n" + "=" * 60)
print("DIVISÃO DOS DADOS")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Tamanho do conjunto de treino: {X_train.shape[0]} amostras")
print(f"📊 Tamanho do conjunto de teste: {X_test.shape[0]} amostras")
print(f"\nDistribuição no treino:")
print(f"  Sobreviveram: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"  Não sobreviveram: {(~y_train.astype(bool)).sum()} ({(1-y_train.mean())*100:.2f}%)")

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✅ Dados normalizados com StandardScaler")

# ==========================
# 5. TREINAMENTO DOS MODELOS
# ==========================
print("\n" + "=" * 60)
print("TREINAMENTO DOS MODELOS DE CLASSIFICAÇÃO")
print("=" * 60)

# Dicionário de modelos
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n🚀 Treinando {name}...")
    
    # Treinar modelo
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'predictions': y_pred,
        'predictions_proba': y_pred_proba,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"✅ Acurácia: {accuracy*100:.2f}%")
    print(f"✅ Precisão: {precision*100:.2f}%")
    print(f"✅ Recall: {recall*100:.2f}%")
    print(f"✅ F1-Score: {f1:.4f}")
    print(f"📊 Cross-validation Accuracy (5-fold): {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
    print(f"\n📊 Matriz de Confusão:")
    print(cm)

# ==========================
# 6. AVALIAÇÃO DOS MODELOS
# ==========================
print("\n" + "=" * 60)
print("AVALIAÇÃO E COMPARAÇÃO DOS MODELOS")
print("=" * 60)

# Comparar modelos
print("\n📊 Resumo dos Resultados:")
print("-" * 90)
print(f"{'Modelo':<20} {'Acurácia':<12} {'Precisão':<12} {'Recall':<12} {'F1-Score':<12}")
print("-" * 90)
for name, result in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
    print(f"{name:<20} {result['accuracy']*100:<11.2f}% {result['precision']*100:<11.2f}% {result['recall']*100:<11.2f}% {result['f1']:<12.4f}")

# Melhor modelo
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
best_predictions = results[best_model_name]['predictions']

print(f"\n🏆 Melhor modelo: {best_model_name}")
print(f"🎯 Acurácia: {results[best_model_name]['accuracy']*100:.2f}%")
print(f"🎯 Precisão: {results[best_model_name]['precision']*100:.2f}%")
print(f"🎯 Recall: {results[best_model_name]['recall']*100:.2f}%")
print(f"🎯 F1-Score: {results[best_model_name]['f1']:.4f}")

# Relatório detalhado
print(f"\n📊 Relatório de Classificação ({best_model_name}):")
print("-" * 60)
print(classification_report(y_test, best_predictions, target_names=['Não Sobreviveu', 'Sobreviveu']))

# Importância das features (se disponível)
if hasattr(best_model, 'feature_importances_'):
    print(f"\n📊 Importância das Features ({best_model_name}):")
    print("-" * 60)
    importance_df = pd.DataFrame({
        'Feature': features_to_use,
        'Importância': best_model.feature_importances_
    }).sort_values('Importância', ascending=False)
    print(importance_df.to_string(index=False))

# ==========================
# 7. EXPORTAR O MODELO
# ==========================
print("\n" + "=" * 60)
print("EXPORTAR MODELO E SCALER")
print("=" * 60)

import joblib

# Exportar o melhor modelo
joblib.dump(best_model, 'modelo_titanic_survival.pkl')
print(f"✅ Modelo exportado: modelo_titanic_survival.pkl")

# Exportar o scaler
joblib.dump(scaler, 'scaler_titanic_survival.pkl')
print(f"✅ Scaler exportado: scaler_titanic_survival.pkl")

# Exportar informações adicionais
model_info = {
    'model_name': best_model_name,
    'features': features_to_use,
    'accuracy': results[best_model_name]['accuracy'],
    'precision': results[best_model_name]['precision'],
    'recall': results[best_model_name]['recall'],
    'f1_score': results[best_model_name]['f1']
}
joblib.dump(model_info, 'model_info_survival.pkl')
print(f"✅ Informações do modelo exportadas: model_info_survival.pkl")

print("\n💡 Para carregar o modelo depois:")
print("   modelo = joblib.load('modelo_titanic_survival.pkl')")
print("   scaler = joblib.load('scaler_titanic_survival.pkl')")
print("   info = joblib.load('model_info_survival.pkl')")

# ==========================
# 8. CONCLUSÕES
# ==========================
print("\n" + "=" * 60)
print("CONCLUSÕES")
print("=" * 60)
print(f"""
✅ Modelo de classificação treinado com sucesso!
🎯 Melhor modelo: {best_model_name}
📊 Acurácia: {results[best_model_name]['accuracy']*100:.2f}%
📊 Precisão: {results[best_model_name]['precision']*100:.2f}%
📊 Recall: {results[best_model_name]['recall']*100:.2f}%
📊 F1-Score: {results[best_model_name]['f1']:.4f}
🔍 Total de passageiros analisados: {len(df_processed)}
📈 Cross-validation Accuracy: {results[best_model_name]['cv_mean']*100:.2f}%

💡 O modelo pode prever se um passageiro sobreviveu ou não
   com base em características como classe, sexo, idade,
   tarifa, família e porto de embarque.
   
📌 Interpretação da Acurácia: O modelo acerta 
   {results[best_model_name]['accuracy']*100:.2f}% das predições.
""")

print("=" * 60)
print("ANÁLISE CONCLUÍDA!")
print("=" * 60)