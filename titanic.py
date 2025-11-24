"""
Interface Gráfica para Predição de Sobrevivência no Titanic
Sistema de Aprendizado Supervisionado com Interface Interativa
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import os
import urllib.request
warnings.filterwarnings('ignore')


class TitanicPredictor:
    """Classe para gerenciar o modelo de predição"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                        'Embarked', 'FamilySize', 'IsAlone', 'Has_Cabin', 'AgeGroup']
        self.model_accuracy = 0
        self.feature_importance = None
        
    def preprocess_data(self, df):
        """Pré-processar os dados do dataset"""
        df_processed = df.copy()
        
        # Preencher valores ausentes
        df_processed['Age'] = df_processed['Age'].fillna(df_processed['Age'].median())
        df_processed['Embarked'] = df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0])
        df_processed['Has_Cabin'] = df_processed['Cabin'].notna().astype(int)
        df_processed['Fare'] = df_processed['Fare'].fillna(df_processed['Fare'].median())
        
        # Criar novas features
        df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
        df_processed['IsAlone'] = (df_processed['FamilySize'] == 1).astype(int)
        df_processed['AgeGroup'] = pd.cut(df_processed['Age'], 
                                          bins=[0, 12, 18, 35, 60, 100],
                                          labels=[0, 1, 2, 3, 4])
        
        # Codificar variáveis categóricas
        le = LabelEncoder()
        df_processed['Sex'] = le.fit_transform(df_processed['Sex'])
        df_processed['Embarked'] = le.fit_transform(df_processed['Embarked'])
        
        return df_processed
    
    def download_titanic_dataset(self):
        """Baixar o dataset do Titanic automaticamente"""
        try:
            url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
            filename = "titanic.csv"
            
            print("Baixando dataset do Titanic...")
            urllib.request.urlretrieve(url, filename)
            print(f"Dataset salvo como: {filename}")
            return filename
        except Exception as e:
            raise Exception(f"Erro ao baixar dataset: {str(e)}")
    
    def train_model(self, filepath=None):
        """Treinar o modelo com o dataset"""
        try:
            # Se não foi fornecido arquivo, tentar baixar
            if filepath is None or not os.path.exists(filepath):
                filepath = self.download_titanic_dataset()
            
            # Carregar dados
            df = pd.read_csv(filepath)
            
            # Verificar se o dataset tem as colunas necessárias
            required_columns = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise Exception(f"Colunas obrigatórias ausentes no dataset: {missing_columns}")
            
            # Pré-processar
            df_processed = self.preprocess_data(df)
            
            # Preparar dados
            X = df_processed[self.features]
            y = df_processed['Survived']
            
            # Dividir em treino e teste
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Treinar modelo
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Avaliar
            y_pred = self.model.predict(X_test)
            self.model_accuracy = accuracy_score(y_test, y_pred)
            
            # Guardar importância das features
            self.feature_importance = dict(zip(self.features, self.model.feature_importances_))
            
            return self.model_accuracy, classification_report(y_test, y_pred), confusion_matrix(y_test, y_pred)
            
        except Exception as e:
            raise Exception(f"Erro no treinamento: {str(e)}")
    
    def predict(self, passenger_data):
        """Fazer predição para um passageiro"""
        if self.model is None:
            raise Exception("Modelo não foi treinado ainda!")
        
        # Converter para DataFrame
        df = pd.DataFrame([passenger_data])
        
        # Fazer predição
        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0]
        
        return prediction, probability


class TitanicApp:
    """Interface Gráfica Principal"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Predição de Sobrevivência - Titanic")
        self.root.geometry("900x700")
        self.root.resizable(False, False)
        
        # Inicializar preditor
        self.predictor = TitanicPredictor()
        self.model_trained = False
        
        # Configurar interface
        self.setup_ui()
        
    def setup_ui(self):
        """Configurar todos os elementos da interface"""
        
        # ==== TÍTULO ====
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        title_frame.pack(fill="x")
        
        title_label = tk.Label(
            title_frame, 
            text="🚢 SISTEMA DE PREDIÇÃO - TITANIC",
            font=("Arial", 20, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack(pady=20)
        
        # ==== CONTAINER PRINCIPAL ====
        main_container = tk.Frame(self.root, bg="#ecf0f1")
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # ==== SEÇÃO DE TREINAMENTO ====
        training_frame = tk.LabelFrame(
            main_container,
            text="📚 1. Carregar e Treinar Modelo",
            font=("Arial", 12, "bold"),
            bg="#ecf0f1",
            fg="#2c3e50"
        )
        training_frame.pack(fill="x", pady=(0, 10))
        
        tk.Label(
            training_frame,
            text="Arquivo do Dataset:",
            font=("Arial", 10),
            bg="#ecf0f1"
        ).grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        self.file_entry = tk.Entry(training_frame, width=40, font=("Arial", 10))
        self.file_entry.insert(0, "Titanic-Dataset.csv")
        self.file_entry.grid(row=0, column=1, padx=10, pady=10)
        
        self.train_button = tk.Button(
            training_frame,
            text="Treinar Modelo",
            command=self.train_model,
            bg="#27ae60",
            fg="white",
            font=("Arial", 10, "bold"),
            cursor="hand2",
            width=15
        )
        self.train_button.grid(row=0, column=2, padx=10, pady=10)
        
        self.status_label = tk.Label(
            training_frame,
            text="⚠️ Modelo não treinado",
            font=("Arial", 10, "bold"),
            bg="#ecf0f1",
            fg="#e74c3c"
        )
        self.status_label.grid(row=1, column=0, columnspan=3, pady=(0, 10))
        
        # ==== SEÇÃO DE ENTRADA DE DADOS ====
        input_frame = tk.LabelFrame(
            main_container,
            text="👤 2. Dados do Passageiro",
            font=("Arial", 12, "bold"),
            bg="#ecf0f1",
            fg="#2c3e50"
        )
        input_frame.pack(fill="x", pady=(0, 10))
        
        # Criar campos de entrada em duas colunas
        self.create_input_fields(input_frame)
        
        # ==== SEÇÃO DE PREDIÇÃO ====
        predict_frame = tk.Frame(main_container, bg="#ecf0f1")
        predict_frame.pack(fill="x", pady=(0, 10))
        
        self.predict_button = tk.Button(
            predict_frame,
            text="🔮 FAZER PREDIÇÃO",
            command=self.make_prediction,
            bg="#3498db",
            fg="white",
            font=("Arial", 14, "bold"),
            cursor="hand2",
            height=2,
            state="disabled"
        )
        self.predict_button.pack(fill="x")
        
        # ==== SEÇÃO DE RESULTADOS ====
        result_frame = tk.LabelFrame(
            main_container,
            text="📊 3. Resultado da Predição",
            font=("Arial", 12, "bold"),
            bg="#ecf0f1",
            fg="#2c3e50"
        )
        result_frame.pack(fill="both", expand=True)
        
        self.result_text = scrolledtext.ScrolledText(
            result_frame,
            width=80,
            height=12,
            font=("Courier", 10),
            bg="#ffffff",
            fg="#2c3e50",
            wrap=tk.WORD
        )
        self.result_text.pack(padx=10, pady=10, fill="both", expand=True)
        
    def create_input_fields(self, parent):
        """Criar campos de entrada para os dados do passageiro"""
        
        # Frame para organizar em duas colunas
        fields_frame = tk.Frame(parent, bg="#ecf0f1")
        fields_frame.pack(padx=10, pady=10)
        
        # Coluna 1
        col1 = tk.Frame(fields_frame, bg="#ecf0f1")
        col1.grid(row=0, column=0, padx=20)
        
        # Coluna 2
        col2 = tk.Frame(fields_frame, bg="#ecf0f1")
        col2.grid(row=0, column=1, padx=20)
        
        # Dicionário para armazenar widgets
        self.input_widgets = {}
        
        # COLUNA 1 - Campos
        row = 0
        
        # Classe
        tk.Label(col1, text="Classe (1, 2 ou 3):", font=("Arial", 10), bg="#ecf0f1").grid(row=row, column=0, sticky="w", pady=5)
        self.input_widgets['Pclass'] = ttk.Combobox(col1, values=[1, 2, 3], width=18, state="readonly")
        self.input_widgets['Pclass'].set(3)
        self.input_widgets['Pclass'].grid(row=row, column=1, pady=5)
        row += 1
        
        # Sexo
        tk.Label(col1, text="Sexo:", font=("Arial", 10), bg="#ecf0f1").grid(row=row, column=0, sticky="w", pady=5)
        self.input_widgets['Sex'] = ttk.Combobox(col1, values=[0, 1], width=18, state="readonly")
        self.input_widgets['Sex'].set(0)
        self.input_widgets['Sex'].grid(row=row, column=1, pady=5)
        tk.Label(col1, text="(0=Homem, 1=Mulher)", font=("Arial", 8), bg="#ecf0f1", fg="#7f8c8d").grid(row=row, column=2, sticky="w", padx=5)
        row += 1
        
        # Idade
        tk.Label(col1, text="Idade:", font=("Arial", 10), bg="#ecf0f1").grid(row=row, column=0, sticky="w", pady=5)
        self.input_widgets['Age'] = tk.Entry(col1, width=20, font=("Arial", 10))
        self.input_widgets['Age'].insert(0, "30")
        self.input_widgets['Age'].grid(row=row, column=1, pady=5)
        row += 1
        
        # Irmãos/Cônjuges
        tk.Label(col1, text="Irmãos/Cônjuges (SibSp):", font=("Arial", 10), bg="#ecf0f1").grid(row=row, column=0, sticky="w", pady=5)
        self.input_widgets['SibSp'] = tk.Spinbox(col1, from_=0, to=10, width=18, font=("Arial", 10))
        self.input_widgets['SibSp'].grid(row=row, column=1, pady=5)
        row += 1
        
        # COLUNA 2 - Campos
        row = 0
        
        # Pais/Filhos
        tk.Label(col2, text="Pais/Filhos (Parch):", font=("Arial", 10), bg="#ecf0f1").grid(row=row, column=0, sticky="w", pady=5)
        self.input_widgets['Parch'] = tk.Spinbox(col2, from_=0, to=10, width=18, font=("Arial", 10))
        self.input_widgets['Parch'].grid(row=row, column=1, pady=5)
        row += 1
        
        # Tarifa
        tk.Label(col2, text="Tarifa (Fare):", font=("Arial", 10), bg="#ecf0f1").grid(row=row, column=0, sticky="w", pady=5)
        self.input_widgets['Fare'] = tk.Entry(col2, width=20, font=("Arial", 10))
        self.input_widgets['Fare'].insert(0, "32.0")
        self.input_widgets['Fare'].grid(row=row, column=1, pady=5)
        row += 1
        
        # Porto de Embarque
        tk.Label(col2, text="Porto de Embarque:", font=("Arial", 10), bg="#ecf0f1").grid(row=row, column=0, sticky="w", pady=5)
        self.input_widgets['Embarked'] = ttk.Combobox(col2, values=[0, 1, 2], width=18, state="readonly")
        self.input_widgets['Embarked'].set(2)
        self.input_widgets['Embarked'].grid(row=row, column=1, pady=5)
        tk.Label(col2, text="(0=C, 1=Q, 2=S)", font=("Arial", 8), bg="#ecf0f1", fg="#7f8c8d").grid(row=row, column=2, sticky="w", padx=5)
        row += 1
        
        # Tem Cabine
        tk.Label(col2, text="Tem Cabine:", font=("Arial", 10), bg="#ecf0f1").grid(row=row, column=0, sticky="w", pady=5)
        self.input_widgets['Has_Cabin'] = ttk.Combobox(col2, values=[0, 1], width=18, state="readonly")
        self.input_widgets['Has_Cabin'].set(0)
        self.input_widgets['Has_Cabin'].grid(row=row, column=1, pady=5)
        tk.Label(col2, text="(0=Não, 1=Sim)", font=("Arial", 8), bg="#ecf0f1", fg="#7f8c8d").grid(row=row, column=2, sticky="w", padx=5)
        
    def train_model(self):
        """Treinar o modelo com o dataset"""
        try:
            filepath = self.file_entry.get()
            # Corrigir nome do arquivo para o padrão correto
            if filepath.strip().lower() == "titanicdataset.csv":
                filepath = "Titanic-Dataset.csv"
            
            # Mostrar mensagem de carregamento
            self.status_label.config(text="⏳ Treinando modelo...", fg="#f39c12")
            self.root.update()
            
            # Treinar
            accuracy, report, cm = self.predictor.train_model(filepath)
            
            # Atualizar status
            self.model_trained = True
            self.status_label.config(
                text=f"✅ Modelo treinado! Acurácia: {accuracy*100:.2f}%",
                fg="#27ae60"
            )
            self.predict_button.config(state="normal")
            
            # Mostrar informações do modelo
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "="*70 + "\n")
            self.result_text.insert(tk.END, "INFORMAÇÕES DO MODELO TREINADO\n")
            self.result_text.insert(tk.END, "="*70 + "\n\n")
            self.result_text.insert(tk.END, f"✅ Acurácia do modelo: {accuracy*100:.2f}%\n\n")
            self.result_text.insert(tk.END, "📊 IMPORTÂNCIA DAS FEATURES:\n")
            self.result_text.insert(tk.END, "-"*70 + "\n")
            
            # Ordenar features por importância
            sorted_features = sorted(self.predictor.feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)
            
            for feature, importance in sorted_features:
                self.result_text.insert(tk.END, f"{feature:.<30} {importance:.4f} ({importance*100:.2f}%)\n")
            
            self.result_text.insert(tk.END, "\n" + "="*70 + "\n")
            self.result_text.insert(tk.END, "✨ Modelo pronto para fazer predições!\n")
            self.result_text.insert(tk.END, "👉 Preencha os dados do passageiro e clique em 'FAZER PREDIÇÃO'\n")
            self.result_text.insert(tk.END, "="*70 + "\n")
            
            messagebox.showinfo("Sucesso", f"Modelo treinado com sucesso!\nAcurácia: {accuracy*100:.2f}%")
            
        except FileNotFoundError:
            messagebox.showerror("Erro", f"Arquivo '{filepath}' não encontrado!")
            self.status_label.config(text="❌ Erro ao carregar arquivo", fg="#e74c3c")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao treinar modelo:\n{str(e)}")
            self.status_label.config(text="❌ Erro no treinamento", fg="#e74c3c")
    
    def make_prediction(self):
        """Fazer predição com os dados inseridos"""
        try:
            # Coletar dados
            pclass = int(self.input_widgets['Pclass'].get())
            sex = int(self.input_widgets['Sex'].get())
            age = float(self.input_widgets['Age'].get())
            sibsp = int(self.input_widgets['SibSp'].get())
            parch = int(self.input_widgets['Parch'].get())
            fare = float(self.input_widgets['Fare'].get())
            embarked = int(self.input_widgets['Embarked'].get())
            has_cabin = int(self.input_widgets['Has_Cabin'].get())
            
            # Calcular features derivadas
            family_size = sibsp + parch + 1
            is_alone = 1 if family_size == 1 else 0
            
            # Determinar faixa etária
            if age <= 12:
                age_group = 0
            elif age <= 18:
                age_group = 1
            elif age <= 35:
                age_group = 2
            elif age <= 60:
                age_group = 3
            else:
                age_group = 4
            
            # Preparar dados
            passenger_data = {
                'Pclass': pclass,
                'Sex': sex,
                'Age': age,
                'SibSp': sibsp,
                'Parch': parch,
                'Fare': fare,
                'Embarked': embarked,
                'FamilySize': family_size,
                'IsAlone': is_alone,
                'Has_Cabin': has_cabin,
                'AgeGroup': age_group
            }
            
            # Fazer predição
            prediction, probability = self.predictor.predict(passenger_data)
            
            # Mostrar resultado
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "="*70 + "\n")
            self.result_text.insert(tk.END, "RESULTADO DA PREDIÇÃO\n")
            self.result_text.insert(tk.END, "="*70 + "\n\n")
            
            # Dados do passageiro
            self.result_text.insert(tk.END, "👤 DADOS DO PASSAGEIRO:\n")
            self.result_text.insert(tk.END, "-"*70 + "\n")
            self.result_text.insert(tk.END, f"Classe: {pclass}ª classe\n")
            self.result_text.insert(tk.END, f"Sexo: {'Mulher' if sex == 1 else 'Homem'}\n")
            self.result_text.insert(tk.END, f"Idade: {age} anos\n")
            self.result_text.insert(tk.END, f"Irmãos/Cônjuges: {sibsp}\n")
            self.result_text.insert(tk.END, f"Pais/Filhos: {parch}\n")
            self.result_text.insert(tk.END, f"Tamanho da Família: {family_size}\n")
            self.result_text.insert(tk.END, f"Está sozinho: {'Sim' if is_alone else 'Não'}\n")
            self.result_text.insert(tk.END, f"Tarifa: ${fare:.2f}\n")
            self.result_text.insert(tk.END, f"Porto: {'Cherbourg' if embarked == 0 else 'Queenstown' if embarked == 1 else 'Southampton'}\n")
            self.result_text.insert(tk.END, f"Tem Cabine: {'Sim' if has_cabin else 'Não'}\n\n")
            
            # Resultado da predição
            self.result_text.insert(tk.END, "🔮 PREDIÇÃO:\n")
            self.result_text.insert(tk.END, "="*70 + "\n")
            
            if prediction == 1:
                self.result_text.insert(tk.END, "✅ SOBREVIVERIA!\n\n", "survived")
                result_msg = "Este passageiro teria SOBREVIVIDO ao naufrágio do Titanic!"
            else:
                self.result_text.insert(tk.END, "❌ NÃO SOBREVIVERIA\n\n", "died")
                result_msg = "Este passageiro NÃO teria sobrevivido ao naufrágio do Titanic."
            
            # Configurar tags de cor
            self.result_text.tag_config("survived", foreground="#27ae60", font=("Arial", 14, "bold"))
            self.result_text.tag_config("died", foreground="#e74c3c", font=("Arial", 14, "bold"))
            
            # Probabilidades
            self.result_text.insert(tk.END, f"📊 Probabilidade de NÃO sobreviver: {probability[0]*100:.2f}%\n")
            self.result_text.insert(tk.END, f"📊 Probabilidade de SOBREVIVER: {probability[1]*100:.2f}%\n\n")
            
            # Confiança
            confidence = max(probability) * 100
            self.result_text.insert(tk.END, f"🎯 Confiança da predição: {confidence:.2f}%\n")
            
            self.result_text.insert(tk.END, "\n" + "="*70 + "\n")
            
            # Mostrar popup com resultado
            messagebox.showinfo(
                "Resultado da Predição",
                f"{result_msg}\n\nConfiança: {confidence:.2f}%"
            )
            
        except ValueError:
            messagebox.showerror("Erro", "Por favor, preencha todos os campos corretamente!")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao fazer predição:\n{str(e)}")


# ==== EXECUTAR APLICAÇÃO ====
if __name__ == "__main__":
    root = tk.Tk()
    app = TitanicApp(root)
    root.mainloop()