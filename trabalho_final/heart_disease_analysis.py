"""
Sistema Completo de Análise e Predição de Risco Cardíaco
Análise Exploratória + Treinamento de Modelos + Exportação
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, 
                             roc_auc_score, roc_curve)
import warnings
import os
from pathlib import Path
import joblib
warnings.filterwarnings('ignore')

# ==========================
# FUNÇÃO PARA CARREGAR DATASET
# ==========================
def load_dataset(candidates=None):
    """Carregar dataset de forma robusta"""
    base = Path(__file__).resolve().parent
    if candidates is None:
        candidates = [
            'heart_disease.csv',
            'Heart_Disease.csv',
            'heart-disease.csv',
            'heartdisease.csv',
        ]
    
    # 1) procurar na mesma pasta do script
    for name in candidates:
        p = base / name
        if p.exists():
            print(f"✅ Dataset encontrado: {p}")
            return pd.read_csv(p)
    
    # 2) procurar no cwd
    for name in candidates:
        p = Path.cwd() / name
        if p.exists():
            print(f"✅ Dataset encontrado: {p}")
            return pd.read_csv(p)
    
    # 3) procurar qualquer CSV com 'heart' no nome
    for p in list(base.glob('*.csv')) + list(Path.cwd().glob('*.csv')):
        if 'heart' in p.name.lower():
            print(f"✅ Dataset encontrado: {p}")
            return pd.read_csv(p)
    
    # 4) falha com mensagem útil
    files_script = ', '.join([f.name for f in base.iterdir() if f.is_file()])
    files_cwd = ', '.join([f.name for f in Path.cwd().iterdir() if f.is_file()])
    raise FileNotFoundError(
        f"Dataset não encontrado. Procurei em:\n  script dir: {base}\n  cwd: {Path.cwd()}\n\n"
        f"Arquivos no script dir: {files_script}\nArquivos no cwd: {files_cwd}\n\n"
        "Coloque o arquivo 'heart_disease.csv' na pasta do projeto."
    )

# ==========================
# 1. CARREGAR DADOS
# ==========================
print("=" * 70)
print("SISTEMA DE ANÁLISE E PREDIÇÃO DE RISCO CARDÍACO")
print("=" * 70)

df = load_dataset()

print("\n📊 Primeiras linhas do dataset:")
print(df.head(10))

print(f"\n📏 Dimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")

# ==========================
# 2. ANÁLISE EXPLORATÓRIA
# ==========================
print("\n" + "=" * 70)
print("ANÁLISE EXPLORATÓRIA DOS DADOS")
print("=" * 70)

print("\n🔍 Informações do dataset:")
print(df.info())

print("\n📈 Estatísticas descritivas (features numéricas):")
print(df.describe())

print("\n❓ Valores nulos por coluna:")
null_counts = df.isnull().sum()
print(null_counts[null_counts > 0])
print(f"\nTotal de valores nulos: {df.isnull().sum().sum()}")

# Verificar a variável alvo
target_col = 'Heart Disease Status'
if target_col in df.columns:
    print(f"\n📊 Distribuição da variável alvo ('{target_col}'):")
    print(df[target_col].value_counts())
    
    # Calcular taxa
    if df[target_col].dtype == 'object':
        yes_count = (df[target_col] == 'Yes').sum()
        total = len(df[df[target_col].notna()])
        print(f"\nTaxa de doença cardíaca: {yes_count}/{total} = {yes_count/total*100:.2f}%")
    else:
        print(f"\nTaxa de doença cardíaca: {df[target_col].mean()*100:.2f}%")
else:
    print(f"\n⚠️ Coluna alvo '{target_col}' não encontrada!")
    print(f"Colunas disponíveis: {list(df.columns)}")

# ==========================
# 3. PRÉ-PROCESSAMENTO
# ==========================
print("\n" + "=" * 70)
print("PRÉ-PROCESSAMENTO DOS DADOS")
print("=" * 70)

df_processed = df.copy()

# Remover linhas onde a variável alvo é nula
print(f"\n🔧 Removendo linhas com variável alvo nula...")
print(f"Linhas antes: {len(df_processed)}")
df_processed = df_processed[df_processed[target_col].notna()]
print(f"Linhas depois: {len(df_processed)}")

# Tratar valores nulos nas features
print("\n🔧 Tratando valores nulos nas features...")
for col in df_processed.columns:
    if col == target_col:
        continue
    
    null_count = df_processed[col].isnull().sum()
    if null_count > 0:
        if df_processed[col].dtype in ['float64', 'int64']:
            # Preencher com mediana para numéricos
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
            print(f"  • {col}: {null_count} nulos preenchidos com mediana")
        else:
            # Preencher com moda para categóricos
            mode_val = df_processed[col].mode()
            if len(mode_val) > 0:
                df_processed[col].fillna(mode_val[0], inplace=True)
                print(f"  • {col}: {null_count} nulos preenchidos com moda ({mode_val[0]})")

print(f"\nValores nulos restantes: {df_processed.isnull().sum().sum()}")

# Codificar variável alvo
print("\n🔧 Codificando variável alvo...")
if df_processed[target_col].dtype == 'object':
    le_target = LabelEncoder()
    df_processed[target_col] = le_target.fit_transform(df_processed[target_col])
    print(f"  Classes: {list(le_target.classes_)} -> {list(range(len(le_target.classes_)))}")

# Codificar variáveis categóricas
print("\n🔧 Codificando variáveis categóricas...")
categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}

for col in categorical_cols:
    if col != target_col:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
        print(f"  • {col}: {len(le.classes_)} classes únicas")

# Separar features e target
X = df_processed.drop(columns=[target_col])
y = df_processed[target_col]

print(f"\n✅ Features: {X.shape[1]} colunas")
print(f"✅ Target: {target_col}")
print(f"✅ Amostras totais: {X.shape[0]}")
print(f"\nDistribuição do target:")
print(f"  Classe 0 (Não): {(y == 0).sum()} ({(y == 0).mean()*100:.2f}%)")
print(f"  Classe 1 (Sim): {(y == 1).sum()} ({(y == 1).mean()*100:.2f}%)")

# ==========================
# 4. DIVISÃO TREINO/TESTE
# ==========================
print("\n" + "=" * 70)
print("DIVISÃO DOS DADOS")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Conjunto de treino: {X_train.shape[0]} amostras")
print(f"📊 Conjunto de teste: {X_test.shape[0]} amostras")
print(f"\nDistribuição no treino:")
print(f"  Classe 0: {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.2f}%)")
print(f"  Classe 1: {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.2f}%)")

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✅ Dados normalizados com StandardScaler")

# ==========================
# 5. TREINAMENTO DOS MODELOS
# ==========================
print("\n" + "=" * 70)
print("TREINAMENTO DOS MODELOS DE CLASSIFICAÇÃO")
print("=" * 70)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42)
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
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    results[name] = {
        'model': model,
        'predictions': y_pred,
        'predictions_proba': y_pred_proba,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"  ✅ Acurácia: {accuracy*100:.2f}%")
    print(f"  ✅ Precisão: {precision*100:.2f}%")
    print(f"  ✅ Recall: {recall*100:.2f}%")
    print(f"  ✅ F1-Score: {f1:.4f}")
    if roc_auc:
        print(f"  ✅ ROC AUC: {roc_auc:.4f}")
    print(f"  📊 CV Accuracy (5-fold): {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")

# ==========================
# 6. AVALIAÇÃO E COMPARAÇÃO
# ==========================
print("\n" + "=" * 70)
print("AVALIAÇÃO E COMPARAÇÃO DOS MODELOS")
print("=" * 70)

print("\n📊 Resumo dos Resultados:")
print("-" * 100)
print(f"{'Modelo':<20} {'Acurácia':<12} {'Precisão':<12} {'Recall':<12} {'F1-Score':<12} {'ROC AUC':<12}")
print("-" * 100)
for name, result in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
    roc_auc_str = f"{result['roc_auc']:.4f}" if result['roc_auc'] else "N/A"
    print(f"{name:<20} {result['accuracy']*100:<11.2f}% {result['precision']*100:<11.2f}% "
          f"{result['recall']*100:<11.2f}% {result['f1']:<12.4f} {roc_auc_str:<12}")

# Melhor modelo
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
best_predictions = results[best_model_name]['predictions']

print(f"\n🏆 MELHOR MODELO: {best_model_name}")
print(f"  🎯 Acurácia: {results[best_model_name]['accuracy']*100:.2f}%")
print(f"  🎯 Precisão: {results[best_model_name]['precision']*100:.2f}%")
print(f"  🎯 Recall: {results[best_model_name]['recall']*100:.2f}%")
print(f"  🎯 F1-Score: {results[best_model_name]['f1']:.4f}")
if results[best_model_name]['roc_auc']:
    print(f"  🎯 ROC AUC: {results[best_model_name]['roc_auc']:.4f}")

# Relatório detalhado
print(f"\n📊 Relatório de Classificação ({best_model_name}):")
print("-" * 70)
print(classification_report(y_test, best_predictions, target_names=['Sem Doença', 'Com Doença']))

# Matriz de confusão
print(f"\n📊 Matriz de Confusão ({best_model_name}):")
print(results[best_model_name]['confusion_matrix'])
tn, fp, fn, tp = results[best_model_name]['confusion_matrix'].ravel()
print(f"\n  Verdadeiros Negativos (TN): {tn}")
print(f"  Falsos Positivos (FP): {fp}")
print(f"  Falsos Negativos (FN): {fn}")
print(f"  Verdadeiros Positivos (TP): {tp}")

# Importância das features (se disponível)
if hasattr(best_model, 'feature_importances_'):
    print(f"\n📊 Importância das Features (Top 15 - {best_model_name}):")
    print("-" * 70)
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importância': best_model.feature_importances_
    }).sort_values('Importância', ascending=False)
    print(importance_df.head(15).to_string(index=False))

# ==========================
# 7. EXPORTAR MODELO E ARTEFATOS
# ==========================
print("\n" + "=" * 70)
print("EXPORTAR MODELO E ARTEFATOS")
print("=" * 70)

# Criar diretório para modelos se não existir
models_dir = Path(__file__).resolve().parent / 'models'
models_dir.mkdir(exist_ok=True)

# Exportar o melhor modelo
model_path = models_dir / 'heart_disease_model.pkl'
joblib.dump(best_model, model_path)
print(f"✅ Modelo exportado: {model_path}")

# Exportar o scaler
scaler_path = models_dir / 'heart_disease_scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler exportado: {scaler_path}")

# Exportar informações adicionais
model_info = {
    'model_name': best_model_name,
    'feature_names': list(X.columns),
    'target_name': target_col,
    'accuracy': results[best_model_name]['accuracy'],
    'precision': results[best_model_name]['precision'],
    'recall': results[best_model_name]['recall'],
    'f1_score': results[best_model_name]['f1'],
    'roc_auc': results[best_model_name]['roc_auc'],
    'label_encoders': label_encoders,
    'cv_mean': results[best_model_name]['cv_mean'],
    'cv_std': results[best_model_name]['cv_std']
}

info_path = models_dir / 'heart_disease_model_info.pkl'
joblib.dump(model_info, info_path)
print(f"✅ Informações do modelo exportadas: {info_path}")

# Exportar todos os resultados
all_results_path = models_dir / 'all_models_results.pkl'
joblib.dump(results, all_results_path)
print(f"✅ Resultados de todos os modelos: {all_results_path}")

print("\n💡 Para carregar o modelo depois:")
print(f"   modelo = joblib.load('{model_path}')")
print(f"   scaler = joblib.load('{scaler_path}')")
print(f"   info = joblib.load('{info_path}')")

# ==========================
# 8. CONCLUSÕES
# ==========================
print("\n" + "=" * 70)
print("CONCLUSÕES")
print("=" * 70)
print(f"""
✅ Sistema de predição de risco cardíaco treinado com sucesso!

🏆 MELHOR MODELO: {best_model_name}
📊 Métricas de Desempenho:
   • Acurácia: {results[best_model_name]['accuracy']*100:.2f}%
   • Precisão: {results[best_model_name]['precision']*100:.2f}%
   • Recall: {results[best_model_name]['recall']*100:.2f}%
   • F1-Score: {results[best_model_name]['f1']:.4f}
   • ROC AUC: {results[best_model_name]['roc_auc']:.4f if results[best_model_name]['roc_auc'] else 'N/A'}
   • CV Accuracy: {results[best_model_name]['cv_mean']*100:.2f}% (±{results[best_model_name]['cv_std']*100:.2f}%)

🔍 Total de pacientes analisados: {len(df_processed)}
📈 Features utilizadas: {X.shape[1]}

💡 O modelo pode prever o risco de doença cardíaca com base em:
   • Dados demográficos (idade, sexo)
   • Indicadores clínicos (pressão arterial, colesterol, diabetes, IMC)
   • Hábitos de vida (exercício, tabagismo, álcool, sono)
   • Histórico familiar e outros fatores de risco

📌 Interpretação:
   • Acurácia de {results[best_model_name]['accuracy']*100:.2f}%: O modelo acerta {results[best_model_name]['accuracy']*100:.2f}% das predições
   • Recall de {results[best_model_name]['recall']*100:.2f}%: Detecta {results[best_model_name]['recall']*100:.2f}% dos casos reais de doença
   • Precisão de {results[best_model_name]['precision']*100:.2f}%: Quando prevê doença, está correto {results[best_model_name]['precision']*100:.2f}% das vezes

🎯 Próximos passos:
   1. Use a interface gráfica (heart_disease_gui.py) para fazer predições interativas
   2. O modelo está salvo em: {models_dir}
   3. Pode ser integrado em aplicações médicas ou sistemas de triagem
""")

print("=" * 70)
print("ANÁLISE CONCLUÍDA!")
print("=" * 70)
