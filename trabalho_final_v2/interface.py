"""
Interface Gráfica para Detector de Placas de Veículos
Projeto para Cadeira de Inteligência Artificial
Autor: [João Alberto José Simango]
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import os
from pathlib import Path
from detector import DetectorPlacas


class InterfaceDetector:
    """Interface gráfica para o detector de placas"""

    def __init__(self):
        """Inicializa a interface"""
        self.root = tk.Tk()
        self.root.title("🚗 Detector de Placas de Veículos")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)

        # Inicializar detector
        self.detector = DetectorPlacas()

        # Variáveis
        self.imagem_original = None
        self.imagem_resultado = None
        self.caminho_imagem = None
        self.texto_placa = ""
        
        # Guardar referências das PhotoImages para evitar garbage collection
        self.photo_original = None
        self.photo_resultado = None

        # Criar interface
        self.criar_widgets()

        # Configurar estilo
        self.configurar_estilo()

    def configurar_estilo(self):
        """Configura o estilo da interface"""
        style = ttk.Style()

        # Configurar cores e fontes
        self.root.configure(bg='#f0f0f0')

        # Estilo dos botões
        style.configure('TButton',
                       font=('Arial', 10, 'bold'),
                       padding=10)

        style.configure('Success.TButton',
                       background='#28a745',
                       foreground='white')

        style.configure('Primary.TButton',
                       background='#007bff',
                       foreground='white')

    def criar_widgets(self):
        """Cria todos os widgets da interface"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Título
        titulo_label = ttk.Label(
            main_frame,
            text="🚗 Detector de Placas de Veículos",
            font=('Arial', 16, 'bold')
        )
        titulo_label.pack(pady=10)

        # Frame para controles
        controles_frame = ttk.Frame(main_frame)
        controles_frame.pack(fill=tk.X, pady=10)

        # Botão para escolher imagem
        self.btn_escolher = ttk.Button(
            controles_frame,
            text="📁 Escolher Imagem",
            command=self.escolher_imagem,
            style='Primary.TButton'
        )
        self.btn_escolher.pack(side=tk.LEFT, padx=5)

        # Botão para processar
        self.btn_processar = ttk.Button(
            controles_frame,
            text="🔍 Processar Imagem",
            command=self.processar_imagem,
            state=tk.DISABLED
        )
        self.btn_processar.pack(side=tk.LEFT, padx=5)

        # Botão para salvar resultado
        self.btn_salvar = ttk.Button(
            controles_frame,
            text="💾 Salvar Resultado",
            command=self.salvar_resultado,
            state=tk.DISABLED
        )
        self.btn_salvar.pack(side=tk.LEFT, padx=5)

        # Label para mostrar caminho da imagem
        self.caminho_label = ttk.Label(
            controles_frame,
            text="Nenhuma imagem selecionada",
            font=('Arial', 9)
        )
        self.caminho_label.pack(side=tk.LEFT, padx=20, fill=tk.X, expand=True)

        # Frame para imagens
        imagens_frame = ttk.Frame(main_frame)
        imagens_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Frame para imagem original
        original_frame = ttk.LabelFrame(imagens_frame, text="Imagem Original", padding="5")
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Canvas para imagem original
        self.canvas_original = tk.Canvas(
            original_frame,
            bg='gray',
            width=400,
            height=300
        )
        self.canvas_original.pack(fill=tk.BOTH, expand=True)

        # Frame para imagem resultado
        resultado_frame = ttk.LabelFrame(imagens_frame, text="Resultado", padding="5")
        resultado_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        # Canvas para imagem resultado
        self.canvas_resultado = tk.Canvas(
            resultado_frame,
            bg='gray',
            width=400,
            height=300
        )
        self.canvas_resultado.pack(fill=tk.BOTH, expand=True)

        # Frame para informações
        info_frame = ttk.LabelFrame(main_frame, text="Informações da Placa", padding="10")
        info_frame.pack(fill=tk.X, pady=10)

        # Label para texto da placa
        ttk.Label(info_frame, text="Texto Detectado:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)

        self.texto_label = ttk.Label(
            info_frame,
            text="Nenhuma placa detectada ainda",
            font=('Courier', 12),
            background='#f8f9fa',
            relief='sunken',
            padding=5
        )
        self.texto_label.pack(fill=tk.X, pady=5)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Pronto")
        status_bar = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            relief='sunken',
            padding=5
        )
        status_bar.pack(fill=tk.X, pady=5)

        # Progress bar (inicialmente oculto)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            main_frame,
            variable=self.progress_var,
            maximum=100
        )
        # Inicialmente oculto
        self.progress_bar.pack(fill=tk.X, pady=5)
        self.progress_bar.pack_forget()

    def escolher_imagem(self):
        """Abre diálogo para escolher imagem"""
        tipos_arquivo = [
            ('Imagens', '*.jpg *.jpeg *.png *.bmp *.tiff'),
            ('Todos os arquivos', '*.*')
        ]

        caminho = filedialog.askopenfilename(
            title="Escolher imagem",
            filetypes=tipos_arquivo
        )

        if caminho:
            self.caminho_imagem = caminho
            self.caminho_label.config(text=f"📁 {os.path.basename(caminho)}")

            # Carregar e mostrar imagem original
            self.carregar_imagem_original()

            # Habilitar botão de processar
            self.btn_processar.config(state=tk.NORMAL)

            # Limpar resultados anteriores
            self.limpar_resultados()

            self.status_var.set("Imagem carregada. Clique em 'Processar Imagem'.")

    def carregar_imagem_original(self):
        """Carrega e mostra a imagem original no canvas"""
        try:
            # Carregar imagem com OpenCV
            self.imagem_original = cv2.imread(self.caminho_imagem)
            if self.imagem_original is None:
                raise ValueError("Não foi possível carregar a imagem")

            # Converter para RGB para PIL
            imagem_rgb = cv2.cvtColor(self.imagem_original, cv2.COLOR_BGR2RGB)

            # Criar imagem PIL
            pil_image = Image.fromarray(imagem_rgb)

            # Redimensionar mantendo proporção
            self.mostrar_imagem_canvas(pil_image, self.canvas_original, 'original')

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar imagem: {e}")
            self.status_var.set("Erro ao carregar imagem")

    def mostrar_imagem_canvas(self, pil_image, canvas, tipo='resultado'):
        """Mostra uma imagem PIL em um canvas tkinter"""
        # Obter dimensões do canvas
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        # Se canvas ainda não foi renderizado, usar valores padrão
        if canvas_width <= 1:
            canvas_width = 400
        if canvas_height <= 1:
            canvas_height = 300

        # Calcular proporção
        img_width, img_height = pil_image.size
        ratio = min(canvas_width / img_width, canvas_height / img_height)

        # Redimensionar
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Converter para PhotoImage e guardar referência
        photo_image = ImageTk.PhotoImage(resized_image)
        
        # Guardar referência baseado no tipo para evitar garbage collection
        if tipo == 'original':
            self.photo_original = photo_image
        else:
            self.photo_resultado = photo_image

        # Limpar canvas e mostrar imagem
        canvas.delete("all")
        canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=photo_image,
            anchor=tk.CENTER
        )

    def limpar_resultados(self):
        """Limpa os resultados anteriores"""
        # Limpar canvas resultado
        self.canvas_resultado.delete("all")

        # Limpar texto
        self.texto_label.config(text="Nenhuma placa detectada ainda")

        # Resetar variáveis
        self.imagem_resultado = None
        self.texto_placa = ""
        self.photo_resultado = None

        # Desabilitar botão salvar
        self.btn_salvar.config(state=tk.DISABLED)

    def processar_imagem(self):
        """Processa a imagem selecionada"""
        if not self.caminho_imagem:
            messagebox.showwarning("Aviso", "Selecione uma imagem primeiro!")
            return

        # Mostrar progress bar
        self.progress_bar.pack(fill=tk.X, pady=5)
        self.progress_var.set(0)
        self.status_var.set("Processando imagem...")

        # Desabilitar botões durante processamento
        self.btn_processar.config(state=tk.DISABLED)
        self.btn_escolher.config(state=tk.DISABLED)

        try:
            # Atualizar progresso
            self.progress_var.set(20)
            self.root.update()

            # Processar imagem
            resultado, texto, sucesso = self.detector.processar_imagem(
                self.caminho_imagem,
                salvar_etapas=False
            )

            self.progress_var.set(80)
            self.root.update()

            # Salvar resultados
            self.imagem_resultado = resultado
            self.texto_placa = texto

            # CORREÇÃO: Recarregar e mostrar imagem original novamente
            self.carregar_imagem_original()

            # Mostrar resultado
            if resultado is not None:
                # Converter para RGB
                resultado_rgb = cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB)
                pil_resultado = Image.fromarray(resultado_rgb)
                self.mostrar_imagem_canvas(pil_resultado, self.canvas_resultado, 'resultado')

            # Mostrar texto
            if sucesso and texto:
                self.texto_label.config(
                    text=f"✅ PLACA DETECTADA: {texto}",
                    foreground='green'
                )
                self.btn_salvar.config(state=tk.NORMAL)
                self.status_var.set("Placa detectada com sucesso!")
            else:
                self.texto_label.config(
                    text="❌ Nenhuma placa detectada",
                    foreground='red'
                )
                self.status_var.set("Nenhuma placa detectada")

            self.progress_var.set(100)

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao processar imagem: {e}")
            self.status_var.set("Erro no processamento")

        finally:
            # Reabilitar botões
            self.btn_processar.config(state=tk.NORMAL)
            self.btn_escolher.config(state=tk.NORMAL)

            # Ocultar progress bar após um tempo
            self.root.after(2000, lambda: self.progress_bar.pack_forget())

    def salvar_resultado(self):
        """Salva o resultado em arquivo"""
        if self.imagem_resultado is None:
            messagebox.showwarning("Aviso", "Nenhum resultado para salvar!")
            return

        # Abrir diálogo para salvar
        tipos_arquivo = [
            ('Imagem PNG', '*.png'),
            ('Imagem JPEG', '*.jpg'),
            ('Imagem BMP', '*.bmp')
        ]

        caminho = filedialog.asksaveasfilename(
            title="Salvar resultado",
            defaultextension=".png",
            filetypes=tipos_arquivo
        )

        if caminho:
            try:
                cv2.imwrite(caminho, self.imagem_resultado)
                messagebox.showinfo("Sucesso", f"Resultado salvo em:\n{caminho}")
                self.status_var.set("Resultado salvo com sucesso")
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao salvar: {e}")

    def executar(self):
        """Executa a interface"""
        self.root.mainloop()


def main():
    """Função principal da interface"""
    print("="*60)
    print("🚗 INTERFACE DETECTOR DE PLACAS DE VEÍCULOS")
    print("   Projeto de Inteligência Artificial")
    print("="*60)

    # Criar e executar interface
    interface = InterfaceDetector()
    interface.executar()


if __name__ == "__main__":
    main()