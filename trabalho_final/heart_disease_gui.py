"""
Interface Gráfica para Predição de Risco Cardíaco
Sistema de Predição Interativo com Interface Tkinter
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class HeartDiseasePredictor:
    """Classe para gerenciar o modelo de predição"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_info = None
        self.models_dir = Path(__file__).resolve().parent / 'models'
        
    def load_model(self):
        """Carregar modelo e artefatos salvos"""
        try:
            model_path = self.models_dir / 'heart_disease_model.pkl'
            scaler_path = self.models_dir / 'heart_disease_scaler.pkl'
            info_path = self.models_dir / 'heart_disease_model_info.pkl'
            
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Modelo não encontrado em: {model_path}\n\n"
                    "Execute primeiro o script 'heart_disease_analysis.py' para treinar o modelo."
                )
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.model_info = joblib.load(info_path)
            
            return True, f"Modelo carregado: {self.model_info['model_name']}"
            
        except Exception as e:
            return False, str(e)
    
    def predict(self, patient_data):
        """Fazer predição para um paciente"""
        if self.model is None:
            raise Exception("Modelo não foi carregado ainda!")
        
        # Criar DataFrame com as features na ordem correta
        df = pd.DataFrame([patient_data], columns=self.model_info['feature_names'])
        
        # Normalizar
        df_scaled = self.scaler.transform(df)
        
        # Fazer predição
        prediction = self.model.predict(df_scaled)[0]
        probability = self.model.predict_proba(df_scaled)[0]
        
        return prediction, probability


class HeartDiseaseApp:
    """Interface Gráfica Principal"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Predição de Risco Cardíaco")
        self.root.geometry("1000x800")
        self.root.resizable(False, False)
        
        # Inicializar preditor
        self.predictor = HeartDiseasePredictor()
        self.model_loaded = False
        
        # Configurar interface
        self.setup_ui()
        
        # Carregar modelo automaticamente
        self.load_model_on_startup()
        
    def setup_ui(self):
        """Configurar todos os elementos da interface"""
        
        # ==== TÍTULO ====
        title_frame = tk.Frame(self.root, bg="#c0392b", height=80)
        title_frame.pack(fill="x")
        
        title_label = tk.Label(
            title_frame, 
            text="❤️ SISTEMA DE PREDIÇÃO DE RISCO CARDÍACO",
            font=("Arial", 20, "bold"),
            bg="#c0392b",
            fg="white"
        )
        title_label.pack(pady=20)
        
        # ==== CONTAINER PRINCIPAL ====
        main_container = tk.Frame(self.root, bg="#ecf0f1")
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # ==== SEÇÃO DE STATUS ====
        status_frame = tk.LabelFrame(
            main_container,
            text="📊 Status do Sistema",
            font=("Arial", 12, "bold"),
            bg="#ecf0f1",
            fg="#2c3e50"
        )
        status_frame.pack(fill="x", pady=(0, 10))
        
        self.status_label = tk.Label(
            status_frame,
            text="⚠️ Modelo não carregado",
            font=("Arial", 10, "bold"),
            bg="#ecf0f1",
            fg="#e74c3c"
        )
        self.status_label.pack(pady=10)
        
        # ==== SEÇÃO DE ENTRADA DE DADOS ====
        input_frame = tk.LabelFrame(
            main_container,
            text="👤 Dados do Paciente",
            font=("Arial", 12, "bold"),
            bg="#ecf0f1",
            fg="#2c3e50"
        )
        input_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        # Criar campos de entrada
        self.create_input_fields(input_frame)
        
        # ==== SEÇÃO DE PREDIÇÃO ====
        predict_frame = tk.Frame(main_container, bg="#ecf0f1")
        predict_frame.pack(fill="x", pady=(0, 10))
        
        button_container = tk.Frame(predict_frame, bg="#ecf0f1")
        button_container.pack()
        
        self.predict_button = tk.Button(
            button_container,
            text="🔮 FAZER PREDIÇÃO",
            command=self.make_prediction,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 14, "bold"),
            cursor="hand2",
            width=20,
            height=2,
            state="disabled"
        )
        self.predict_button.grid(row=0, column=0, padx=5)
        
        clear_button = tk.Button(
            button_container,
            text="🔄 LIMPAR CAMPOS",
            command=self.clear_fields,
            bg="#95a5a6",
            fg="white",
            font=("Arial", 14, "bold"),
            cursor="hand2",
            width=20,
            height=2
        )
        clear_button.grid(row=0, column=1, padx=5)
        
        # ==== SEÇÃO DE RESULTADOS ====
        result_frame = tk.LabelFrame(
            main_container,
            text="📊 Resultado da Predição",
            font=("Arial", 12, "bold"),
            bg="#ecf0f1",
            fg="#2c3e50"
        )
        result_frame.pack(fill="both", expand=True)
        
        self.result_text = scrolledtext.ScrolledText(
            result_frame,
            width=90,
            height=15,
            font=("Courier", 10),
            bg="#ffffff",
            fg="#2c3e50",
            wrap=tk.WORD
        )
        self.result_text.pack(padx=10, pady=10, fill="both", expand=True)
        
    def create_input_fields(self, parent):
        """Criar campos de entrada para os dados do paciente"""
        
        # Canvas com scrollbar para os campos
        canvas_frame = tk.Frame(parent, bg="#ecf0f1")
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        canvas = tk.Canvas(canvas_frame, bg="#ecf0f1", highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#ecf0f1")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Dicionário para armazenar widgets
        self.input_widgets = {}
        
        # Definir campos com suas configurações
        fields_config = [
            # (nome_campo, label, tipo, valores/range, valor_padrão, dica)
            ('Age', 'Idade', 'entry', None, '50', 'anos'),
            ('Gender', 'Sexo', 'combo', ['Male', 'Female'], 'Male', ''),
            ('Blood Pressure', 'Pressão Arterial', 'entry', None, '120', 'mmHg'),
            ('Cholesterol Level', 'Nível de Colesterol', 'entry', None, '200', 'mg/dL'),
            ('Exercise Habits', 'Hábitos de Exercício', 'combo', ['Low', 'Medium', 'High'], 'Medium', ''),
            ('Smoking', 'Fumante', 'combo', ['Yes', 'No'], 'No', ''),
            ('Family Heart Disease', 'Histórico Familiar', 'combo', ['Yes', 'No'], 'No', ''),
            ('Diabetes', 'Diabetes', 'combo', ['Yes', 'No'], 'No', ''),
            ('BMI', 'IMC (Índice de Massa Corporal)', 'entry', None, '25.0', 'kg/m²'),
            ('High Blood Pressure', 'Pressão Alta', 'combo', ['Yes', 'No'], 'No', ''),
            ('Low HDL Cholesterol', 'Colesterol HDL Baixo', 'combo', ['Yes', 'No'], 'No', ''),
            ('High LDL Cholesterol', 'Colesterol LDL Alto', 'combo', ['Yes', 'No'], 'No', ''),
            ('Alcohol Consumption', 'Consumo de Álcool', 'combo', ['None', 'Low', 'Medium', 'High'], 'Low', ''),
            ('Stress Level', 'Nível de Estresse', 'combo', ['Low', 'Medium', 'High'], 'Medium', ''),
            ('Sleep Hours', 'Horas de Sono', 'entry', None, '7.0', 'horas/dia'),
            ('Sugar Consumption', 'Consumo de Açúcar', 'combo', ['Low', 'Medium', 'High'], 'Medium', ''),
            ('Triglyceride Level', 'Nível de Triglicerídeos', 'entry', None, '150', 'mg/dL'),
            ('Fasting Blood Sugar', 'Glicemia em Jejum', 'entry', None, '100', 'mg/dL'),
            ('CRP Level', 'Nível de PCR (Proteína C-Reativa)', 'entry', None, '1.0', 'mg/L'),
            ('Homocysteine Level', 'Nível de Homocisteína', 'entry', None, '10.0', 'µmol/L'),
        ]
        
        # Criar campos em grid de 2 colunas
        for idx, (field_name, label, field_type, values, default, hint) in enumerate(fields_config):
            row = idx // 2
            col = (idx % 2) * 3
            
            # Label
            label_text = f"{label}:"
            if hint:
                label_text += f" ({hint})"
            
            tk.Label(
                scrollable_frame,
                text=label_text,
                font=("Arial", 9),
                bg="#ecf0f1",
                anchor="w"
            ).grid(row=row, column=col, sticky="w", padx=5, pady=5)
            
            # Widget de entrada
            if field_type == 'entry':
                widget = tk.Entry(scrollable_frame, width=15, font=("Arial", 9))
                widget.insert(0, default)
            elif field_type == 'combo':
                widget = ttk.Combobox(scrollable_frame, values=values, width=13, state="readonly")
                widget.set(default)
            
            widget.grid(row=row, column=col+1, padx=5, pady=5, sticky="w")
            self.input_widgets[field_name] = widget
    
    def load_model_on_startup(self):
        """Carregar modelo ao iniciar"""
        success, message = self.predictor.load_model()
        
        if success:
            self.model_loaded = True
            self.status_label.config(
                text=f"✅ {message} | Acurácia: {self.predictor.model_info['accuracy']*100:.2f}%",
                fg="#27ae60"
            )
            self.predict_button.config(state="normal")
            
            # Mostrar informações do modelo
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "="*90 + "\n")
            self.result_text.insert(tk.END, "BEM-VINDO AO SISTEMA DE PREDIÇÃO DE RISCO CARDÍACO\n")
            self.result_text.insert(tk.END, "="*90 + "\n\n")
            self.result_text.insert(tk.END, f"🤖 Modelo: {self.predictor.model_info['model_name']}\n")
            self.result_text.insert(tk.END, f"📊 Acurácia: {self.predictor.model_info['accuracy']*100:.2f}%\n")
            self.result_text.insert(tk.END, f"📊 Precisão: {self.predictor.model_info['precision']*100:.2f}%\n")
            self.result_text.insert(tk.END, f"📊 Recall: {self.predictor.model_info['recall']*100:.2f}%\n")
            self.result_text.insert(tk.END, f"📊 F1-Score: {self.predictor.model_info['f1']:.4f}\n")
            if self.predictor.model_info.get('roc_auc'):
                self.result_text.insert(tk.END, f"📊 ROC AUC: {self.predictor.model_info['roc_auc']:.4f}\n")
            self.result_text.insert(tk.END, f"\n✨ Sistema pronto para fazer predições!\n")
            self.result_text.insert(tk.END, "👉 Preencha os dados do paciente e clique em 'FAZER PREDIÇÃO'\n")
            self.result_text.insert(tk.END, "="*90 + "\n")
        else:
            self.status_label.config(
                text=f"❌ Erro ao carregar modelo: {message}",
                fg="#e74c3c"
            )
            messagebox.showerror("Erro", f"Erro ao carregar modelo:\n\n{message}")
    
    def clear_fields(self):
        """Limpar todos os campos de entrada"""
        defaults = {
            'Age': '50',
            'Gender': 'Male',
            'Blood Pressure': '120',
            'Cholesterol Level': '200',
            'Exercise Habits': 'Medium',
            'Smoking': 'No',
            'Family Heart Disease': 'No',
            'Diabetes': 'No',
            'BMI': '25.0',
            'High Blood Pressure': 'No',
            'Low HDL Cholesterol': 'No',
            'High LDL Cholesterol': 'No',
            'Alcohol Consumption': 'Low',
            'Stress Level': 'Medium',
            'Sleep Hours': '7.0',
            'Sugar Consumption': 'Medium',
            'Triglyceride Level': '150',
            'Fasting Blood Sugar': '100',
            'CRP Level': '1.0',
            'Homocysteine Level': '10.0'
        }
        
        for field_name, widget in self.input_widgets.items():
            if isinstance(widget, tk.Entry):
                widget.delete(0, tk.END)
                widget.insert(0, defaults.get(field_name, ''))
            elif isinstance(widget, ttk.Combobox):
                widget.set(defaults.get(field_name, ''))
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "✅ Campos limpos! Pronto para nova predição.\n")
    
    def make_prediction(self):
        """Fazer predição com os dados inseridos"""
        try:
            # Coletar dados dos campos
            patient_data = {}
            
            for field_name, widget in self.input_widgets.items():
                if isinstance(widget, tk.Entry):
                    value = widget.get().strip()
                    try:
                        patient_data[field_name] = float(value)
                    except ValueError:
                        raise ValueError(f"Valor inválido para '{field_name}': {value}")
                elif isinstance(widget, ttk.Combobox):
                    value = widget.get()
                    # Codificar valores categóricos
                    if field_name in self.predictor.model_info['label_encoders']:
                        le = self.predictor.model_info['label_encoders'][field_name]
                        try:
                            patient_data[field_name] = le.transform([value])[0]
                        except ValueError:
                            # Se o valor não foi visto no treinamento, usar o mais comum
                            patient_data[field_name] = 0
                    else:
                        patient_data[field_name] = value
            
            # Fazer predição
            prediction, probability = self.predictor.predict(list(patient_data.values()))
            
            # Mostrar resultado
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "="*90 + "\n")
            self.result_text.insert(tk.END, "RESULTADO DA PREDIÇÃO DE RISCO CARDÍACO\n")
            self.result_text.insert(tk.END, "="*90 + "\n\n")
            
            # Resultado da predição
            self.result_text.insert(tk.END, "🔮 PREDIÇÃO:\n")
            self.result_text.insert(tk.END, "="*90 + "\n")
            
            if prediction == 1:
                self.result_text.insert(tk.END, "⚠️ ALTO RISCO DE DOENÇA CARDÍACA\n\n", "high_risk")
                result_msg = "⚠️ Este paciente apresenta ALTO RISCO de doença cardíaca!"
                color = "#e74c3c"
            else:
                self.result_text.insert(tk.END, "✅ BAIXO RISCO DE DOENÇA CARDÍACA\n\n", "low_risk")
                result_msg = "✅ Este paciente apresenta BAIXO RISCO de doença cardíaca."
                color = "#27ae60"
            
            # Configurar tags de cor
            self.result_text.tag_config("high_risk", foreground="#e74c3c", font=("Arial", 14, "bold"))
            self.result_text.tag_config("low_risk", foreground="#27ae60", font=("Arial", 14, "bold"))
            
            # Probabilidades
            self.result_text.insert(tk.END, f"📊 Probabilidade de BAIXO RISCO: {probability[0]*100:.2f}%\n")
            self.result_text.insert(tk.END, f"📊 Probabilidade de ALTO RISCO: {probability[1]*100:.2f}%\n\n")
            
            # Confiança
            confidence = max(probability) * 100
            self.result_text.insert(tk.END, f"🎯 Confiança da predição: {confidence:.2f}%\n")
            
            # Interpretação
            self.result_text.insert(tk.END, "\n💡 INTERPRETAÇÃO:\n")
            self.result_text.insert(tk.END, "-"*90 + "\n")
            
            if prediction == 1:
                self.result_text.insert(tk.END, 
                    "⚠️ O paciente apresenta características associadas a maior risco cardiovascular.\n"
                    "   Recomenda-se:\n"
                    "   • Consulta médica especializada\n"
                    "   • Exames cardiológicos complementares\n"
                    "   • Avaliação de fatores de risco modificáveis\n"
                    "   • Possível intervenção preventiva ou terapêutica\n"
                )
            else:
                self.result_text.insert(tk.END,
                    "✅ O paciente apresenta perfil de baixo risco cardiovascular.\n"
                    "   Recomenda-se:\n"
                    "   • Manter hábitos saudáveis\n"
                    "   • Check-ups periódicos\n"
                    "   • Controle contínuo dos fatores de risco\n"
                    "   • Atividade física regular e alimentação balanceada\n"
                )
            
            self.result_text.insert(tk.END, "\n" + "="*90 + "\n")
            self.result_text.insert(tk.END, "⚕️ AVISO: Esta predição é uma ferramenta de apoio à decisão clínica.\n")
            self.result_text.insert(tk.END, "   O diagnóstico definitivo deve ser feito por um profissional de saúde.\n")
            self.result_text.insert(tk.END, "="*90 + "\n")
            
            # Mostrar popup com resultado
            messagebox.showinfo(
                "Resultado da Predição",
                f"{result_msg}\n\nConfiança: {confidence:.2f}%\n\n"
                "Consulte um médico para avaliação completa.",
                icon='warning' if prediction == 1 else 'info'
            )
            
        except ValueError as ve:
            messagebox.showerror("Erro", f"Erro nos dados:\n{str(ve)}")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao fazer predição:\n{str(e)}")


# ==== EXECUTAR APLICAÇÃO ====
if __name__ == "__main__":
    root = tk.Tk()
    app = HeartDiseaseApp(root)
    root.mainloop()
