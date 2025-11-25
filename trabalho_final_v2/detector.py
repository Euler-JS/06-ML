"""
Detector de Placas de Veículos
Projeto para Cadeira de Inteligência Artificial
Autor: [João Alberto José Simango]
"""

import cv2
import numpy as np
import pytesseract
import os
from pathlib import Path
import re


# ========================================
# CLASSE PRINCIPAL
# ========================================
class DetectorPlacas:
    """Classe para detectar e ler placas de veículos"""
    
    def __init__(self):
        """Inicializa o detector com parâmetros padrão"""
        # Parâmetros de detecção de bordas
        self.canny_thresh1 = 50 # Valor inferior para Canny 
        self.canny_thresh2 = 150 # Valor superior para Canny
        
        # Parâmetros de filtragem de contornos (ajustados para imagens pequenas)
        self.min_area = 100 # Área mínima do contorno da placa (reduzido para imagens pequenas)
        self.max_area = 100000 # Área máxima do contorno da placa
        self.min_aspect_ratio = 1.5 # Proporção mínima largura/altura da placa (mais flexível)
        self.max_aspect_ratio = 6.0 # Proporção máxima largura/altura da placa
        
        # Configuração do OCR
        self.tesseract_config = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' # Configuração para reconhecer apenas letras e números
        
        # Contador de imagens processadas
        self.contador = 0 # Contador para nomear imagens salvas
    
    def preprocessar_imagem(self, imagem):
        """
        Pré-processa a imagem para melhorar a detecção
        
        Args:
            imagem: Imagem original em BGR
            
        Returns:
            Imagem pré-processada em escala de cinza
        """
        # Converter para escala de cinza
        gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # Converter para escala de cinza, o que facilita a detecção de bordas e contornos.
        
        # Aplicar filtro bilateral para remover ruído mantendo bordas
        gray = cv2.bilateralFilter(gray, 11, 17, 17) # Aplicar um filtro bilateral para reduzir o ruído na imagem, preservando as bordas importantes.
        
        # Equalizar histograma para melhorar contraste
        gray = cv2.equalizeHist(gray) # Equalizar o histograma da imagem para melhorar o contraste, o que pode ajudar na detecção de bordas.
        
        return gray
    
    def detectar_bordas(self, gray):
        """
        Detecta bordas na imagem usando Canny
        
        Args:
            gray: Imagem em escala de cinza
            
        Returns:
            Imagem com bordas detectadas
        """
        edged = cv2.Canny(gray, self.canny_thresh1, self.canny_thresh2)
        return edged
    
    def encontrar_contornos(self, edged):
        """
        Encontra contornos na imagem de bordas
        
        Args:
            edged: Imagem com bordas detectadas
            
        Returns:
            Lista de contornos ordenados por área (maior para menor)
        """
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Ordenar por área (do maior para o menor)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
        return contours
    
    def filtrar_contorno_placa(self, contours):
        """
        Filtra contornos para encontrar a placa
        
        Args:
            contours: Lista de contornos
            
        Returns:
            Contorno da placa (ou None se não encontrar)
        """
        placa_contorno = None
        
        for contour in contours:
            # Calcular área
            area = cv2.contourArea(contour)
            
            # Filtrar por área
            if area < self.min_area or area > self.max_area:
                continue
            
            # Aproximar contorno para polígono
            perimetro = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * perimetro, True)
            
            # Verificar se tem 4 vértices (retângulo)
            if len(approx) == 4:
                # Calcular bounding box
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                
                # Verificar proporção (placas são retangulares horizontais)
                if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
                    placa_contorno = approx
                    break
        
        return placa_contorno
    
    def extrair_placa(self, imagem, contorno):
        """
        Extrai a região da placa da imagem
        
        Args:
            imagem: Imagem original
            contorno: Contorno da placa
            
        Returns:
            Região da placa recortada
        """
        # Criar máscara
        mask = np.zeros(imagem.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contorno], -1, 255, -1)
        
        # Aplicar máscara
        placa_img = cv2.bitwise_and(imagem, imagem, mask=mask)
        
        # Pegar bounding box
        x, y, w, h = cv2.boundingRect(contorno)
        placa_crop = placa_img[y:y+h, x:x+w]
        
        return placa_crop, (x, y, w, h)
    
    def preparar_para_ocr(self, placa_img):
        """
        Prepara a imagem da placa para OCR com processamento avançado
        
        Args:
            placa_img: Imagem da placa
            
        Returns:
            Imagem processada para OCR
        """
        # Converter para escala de cinza
        gray = cv2.cvtColor(placa_img, cv2.COLOR_BGR2GRAY)
        
        # Redimensionar para tamanho maior ANTES do processamento (melhor para OCR)
        scale = 4  # Aumentado de 3 para 4
        largura = int(gray.shape[1] * scale)
        altura = int(gray.shape[0] * scale)
        resized = cv2.resize(gray, (largura, altura), interpolation=cv2.INTER_CUBIC)
        
        # Aplicar denoising mais forte
        denoised = cv2.fastNlMeansDenoising(resized, None, h=15, templateWindowSize=7, searchWindowSize=21)
        
        # CLAHE para melhor contraste
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        contraste = clahe.apply(denoised)
        
        # Binarização adaptativa com parâmetros ajustados
        binary = cv2.adaptiveThreshold(
            contraste, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 21, 15
        )
        
        # Inverter se fundo for escuro
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)
        
        # Operações morfológicas mais suaves
        kernel = np.ones((2,2), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Adicionar borda branca (ajuda o OCR)
        bordered = cv2.copyMakeBorder(morph, 20, 20, 20, 20, 
                                       cv2.BORDER_CONSTANT, value=255)
        
        return bordered
    
    def _pontuacao_formato(self, texto):
        """Calcula pontuação baseada na proximidade com formato de placa"""
        pontos = 0
        texto_limpo = re.sub(r'[^A-Z0-9]', '', texto.upper())
        
        # Tamanho ideal: 8 caracteres (AAA###AA)
        if len(texto_limpo) == 8:
            pontos += 10
        elif 7 <= len(texto_limpo) <= 9:
            pontos += 5
        
        # Padrão Moçambique: 3 letras + 3 números + 2 letras
        if re.match(r'^[A-Z]{3}[0-9]{3}[A-Z]{2}$', texto_limpo):
            pontos += 20
        elif re.match(r'^[A-Z]{3}[0-9]{3}', texto_limpo):
            pontos += 10
        
        return pontos
    
    def corrigir_confusoes_comuns(self, texto):
        """
        Corrige confusões comuns de OCR baseado na posição dos caracteres
        e padrões de placas moçambicanas (AAA###AA)
        
        Args:
            texto: Texto extraído pelo OCR
            
        Returns:
            Texto corrigido
        """
        if len(texto) < 8:
            return texto
        
        texto_lista = list(texto)
        
        # Posições 0-2: devem ser LETRAS
        for i in range(min(3, len(texto_lista))):
            # Números confundidos com letras
            texto_lista[i] = texto_lista[i].replace('0', 'O')
            texto_lista[i] = texto_lista[i].replace('1', 'I')
            texto_lista[i] = texto_lista[i].replace('5', 'S')
            texto_lista[i] = texto_lista[i].replace('8', 'B')
        
        # Posições 3-5: devem ser NÚMEROS
        for i in range(3, min(6, len(texto_lista))):
            # Letras confundidas com números
            texto_lista[i] = texto_lista[i].replace('O', '0')
            texto_lista[i] = texto_lista[i].replace('I', '1')
            texto_lista[i] = texto_lista[i].replace('Z', '2')
            texto_lista[i] = texto_lista[i].replace('S', '5')
            texto_lista[i] = texto_lista[i].replace('B', '8')
        
        # Posições 6-7: devem ser LETRAS
        for i in range(6, min(8, len(texto_lista))):
            # Números confundidos com letras
            texto_lista[i] = texto_lista[i].replace('0', 'O')
            texto_lista[i] = texto_lista[i].replace('1', 'I')
            texto_lista[i] = texto_lista[i].replace('5', 'S')
            texto_lista[i] = texto_lista[i].replace('8', 'B')
            
            # Correção específica H -> M (H é frequentemente confundido com M)
            # Na primeira letra do sufixo (posição 6), H geralmente é M
            if i == 6 and texto_lista[i] == 'H':
                texto_lista[i] = 'M'
        
        texto_corrigido = ''.join(texto_lista)
        return texto_corrigido
    
    def ler_placa_ocr(self, placa_processada):
        """
        Lê o texto da placa usando OCR com múltiplas tentativas e correções
        
        Args:
            placa_processada: Imagem da placa processada
            
        Returns:
            Texto da placa
        """
        try:
            tentativas = []
            
            # Configurações diferentes para tentar
            configs = [
                '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            ]
            
            # Tentativa 1: Imagem original
            for config in configs:
                texto = pytesseract.image_to_string(placa_processada, config=config)
                texto = texto.strip().upper()
                texto = re.sub(r'[^A-Z0-9]', '', texto)
                if texto:
                    tentativas.append(texto)
            
            # Tentativa 2: Com mais contraste
            contraste = cv2.convertScaleAbs(placa_processada, alpha=1.5, beta=0)
            texto = pytesseract.image_to_string(contraste, config=configs[0])
            texto = texto.strip().upper()
            texto = re.sub(r'[^A-Z0-9]', '', texto)
            if texto:
                tentativas.append(texto)
            
            # Tentativa 3: Com inversão
            invertida = cv2.bitwise_not(placa_processada)
            texto = pytesseract.image_to_string(invertida, config=configs[0])
            texto = texto.strip().upper()
            texto = re.sub(r'[^A-Z0-9]', '', texto)
            if texto:
                tentativas.append(texto)
            
            if not tentativas:
                return ""
            
            # Aplicar correções em todas as tentativas
            tentativas_corrigidas = [self.corrigir_confusoes_comuns(t) for t in tentativas]
            
            # Debug: mostrar tentativas
            print(f"   Tentativas OCR: {tentativas_corrigidas[:3]}")
            
            # Pegar a melhor tentativa (maior pontuação)
            melhor = max(tentativas_corrigidas, key=lambda t: self._pontuacao_formato(t))
            
            return melhor
        except Exception as e:
            print(f"Erro no OCR: {e}")
            return ""
    
    def validar_formato_placa(self, texto):
        """
        Valida se o texto segue formato de placa moçambicana, ignorando símbolo central.
        Aceita:
        - AAA NNN AA (com espaços)
        - AAANNNAA (sem espaços)
        - AAANNNxAA (com símbolo central extraído como caractere)
        Args:
            texto: Texto extraído da placa
        Returns:
            Texto formatado (ou vazio se inválido)
        """
        texto_limpo = re.sub(r'[^A-Z0-9]', '', texto)
        
        # Se vier com 9+ caracteres, tenta identificar e remover caracteres extras
        if len(texto_limpo) > 8:
            # Estratégia: procurar padrão AAA### e pegar os 2 últimos caracteres alfabéticos
            # Remove caracteres extras que geralmente aparecem após os números
            match = re.match(r'^([A-Z]{3})([0-9]{3})(.*)$', texto_limpo)
            if match:
                prefixo = match.group(1)  # AAA
                numeros = match.group(2)  # ###
                resto = match.group(3)    # resto (pode ter caracteres extras)
                
                # Pegar apenas as letras do resto (ignorar números extras)
                letras_finais = re.sub(r'[^A-Z]', '', resto)
                
                # Se temos exatamente 2 letras, usar elas
                if len(letras_finais) >= 2:
                    texto_limpo = prefixo + numeros + letras_finais[:2]
        
        # Só aceita se tiver 8 caracteres
        if len(texto_limpo) != 8:
            return ""
        
        padrao_mocambique = r'^[A-Z]{3}[0-9]{3}[A-Z]{2}$'
        if re.match(padrao_mocambique, texto_limpo):
            return f"{texto_limpo[:3]} {texto_limpo[3:6]} {texto_limpo[6:]}"
        return ""
    
    def desenhar_resultado(self, imagem, contorno, texto, bbox):
        """
        Desenha o resultado na imagem
        
        Args:
            imagem: Imagem original
            contorno: Contorno da placa
            texto: Texto da placa
            bbox: Bounding box (x, y, w, h)
            
        Returns:
            Imagem com anotações
        """
        resultado = imagem.copy()
        
        # Desenhar contorno da placa
        cv2.drawContours(resultado, [contorno], -1, (0, 255, 0), 3)
        
        # Desenhar bounding box
        x, y, w, h = bbox
        cv2.rectangle(resultado, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Adicionar texto
        if texto:
            # Fundo para o texto
            (text_width, text_height), _ = cv2.getTextSize(
                texto, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2
            )
            cv2.rectangle(
                resultado, 
                (x, y - text_height - 10), 
                (x + text_width, y),
                (0, 255, 0), 
                -1
            )
            # Texto
            cv2.putText(
                resultado, texto, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2
            )
        
        return resultado
    
    def eh_imagem_placa_recortada(self, imagem):
        """
        Verifica se a imagem já é uma placa recortada (sem contexto do veículo)
        
        Args:
            imagem: Imagem a verificar
            
        Returns:
            True se for uma placa recortada, False caso contrário
        """
        altura, largura = imagem.shape[:2]
        aspect_ratio = largura / altura
        
        # Placas recortadas geralmente são pequenas e com proporção horizontal
        # Típico: largura entre 2x e 6x a altura
        if altura < 200 and largura < 600 and 2.0 <= aspect_ratio <= 6.0:
            return True
        
        return False
    
    def processar_imagem(self, caminho_imagem, salvar_etapas=False):
        """
        Processa uma imagem completa
        
        Args:
            caminho_imagem: Caminho da imagem
            salvar_etapas: Se deve salvar imagens intermediárias
            
        Returns:
            Tuple (imagem_resultado, texto_placa, sucesso)
        """
        print(f"\n{'='*60}")
        print(f"Processando: {caminho_imagem}")
        print(f"{'='*60}")
        
        # Carregar imagem
        imagem = cv2.imread(caminho_imagem)
        if imagem is None:
            print("❌ Erro ao carregar imagem")
            return None, "", False
        
        print(f"✓ Imagem carregada: {imagem.shape}")
        
        # Verificar se é uma placa já recortada
        if self.eh_imagem_placa_recortada(imagem):
            print("ℹ️  Detectada imagem de placa recortada - processando diretamente...")
            
            # Processar diretamente sem buscar contornos
            print("1️⃣  Preparando para OCR...")
            placa_processada = self.preparar_para_ocr(imagem)
            
            print("2️⃣  Lendo placa com OCR...")
            texto = self.ler_placa_ocr(placa_processada)
            
            print("3️⃣  Validando formato...")
            texto_formatado = self.validar_formato_placa(texto)
            
            if texto_formatado:
                print(f"✅ PLACA LIDA: {texto_formatado}")
            else:
                print(f"⚠️  Texto extraído: {texto} (formato inválido)")
            
            # Criar resultado visual simples
            resultado = imagem.copy()
            if texto_formatado:
                cv2.putText(
                    resultado, texto_formatado, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
            
            # Salvar etapas se solicitado
            if salvar_etapas:
                nome_arquivo = Path(caminho_imagem).stem
                pasta_resultado = Path("resultados") / nome_arquivo
                pasta_resultado.mkdir(parents=True, exist_ok=True)
                
                cv2.imwrite(str(pasta_resultado / "1_original.jpg"), imagem)
                cv2.imwrite(str(pasta_resultado / "2_placa_processada.jpg"), placa_processada)
                cv2.imwrite(str(pasta_resultado / "3_resultado_final.jpg"), resultado)
                print(f"💾 Etapas salvas em: {pasta_resultado}")
            
            return resultado, texto_formatado, bool(texto_formatado)
        
        # Processamento normal para imagens completas
        # 1. Pré-processar
        print("1️⃣  Pré-processando imagem...")
        gray = self.preprocessar_imagem(imagem)
        
        # 2. Detectar bordas
        print("2️⃣  Detectando bordas...")
        edged = self.detectar_bordas(gray)
        
        # 3. Encontrar contornos
        print("3️⃣  Encontrando contornos...")
        contours = self.encontrar_contornos(edged)
        print(f"   Encontrados {len(contours)} contornos")
        
        # 4. Filtrar contorno da placa
        print("4️⃣  Filtrando contorno da placa...")
        placa_contorno = self.filtrar_contorno_placa(contours)
        
        if placa_contorno is None:
            print("❌ Placa não detectada")
            return imagem, "", False
        
        print("✓ Placa detectada!")
        
        # 5. Extrair placa
        print("5️⃣  Extraindo região da placa...")
        placa_img, bbox = self.extrair_placa(imagem, placa_contorno)
        
        # 6. Preparar para OCR
        print("6️⃣  Preparando para OCR...")
        placa_processada = self.preparar_para_ocr(placa_img)
        
        # 7. Ler placa
        print("7️⃣  Lendo placa com OCR...")
        texto = self.ler_placa_ocr(placa_processada)
        
        # 8. Validar formato
        print("8️⃣  Validando formato...")
        texto_formatado = self.validar_formato_placa(texto)
        
        if texto_formatado:
            print(f"✅ PLACA LIDA: {texto_formatado}")
        else:
            print(f"⚠️  Texto extraído: {texto} (formato inválido)")
        
        # 9. Desenhar resultado
        resultado = self.desenhar_resultado(imagem, placa_contorno, texto_formatado, bbox)
        
        # Salvar etapas intermediárias se solicitado
        if salvar_etapas:
            self.salvar_etapas_processamento(
                caminho_imagem, imagem, gray, edged, 
                placa_img, placa_processada, resultado
            )
        
        return resultado, texto_formatado, True
    
    def salvar_etapas_processamento(self, caminho_original, original, gray, edged, 
                                     placa, placa_proc, resultado):
        """Salva imagens intermediárias do processamento"""
        # Criar pasta de resultados
        nome_arquivo = Path(caminho_original).stem
        pasta_resultado = Path("resultados") / nome_arquivo
        pasta_resultado.mkdir(parents=True, exist_ok=True)
        
        # Salvar cada etapa
        cv2.imwrite(str(pasta_resultado / "1_original.jpg"), original)
        cv2.imwrite(str(pasta_resultado / "2_cinza.jpg"), gray)
        cv2.imwrite(str(pasta_resultado / "3_bordas.jpg"), edged)
        cv2.imwrite(str(pasta_resultado / "4_placa_extraida.jpg"), placa)
        cv2.imwrite(str(pasta_resultado / "5_placa_processada.jpg"), placa_proc)
        cv2.imwrite(str(pasta_resultado / "6_resultado_final.jpg"), resultado)
        
        print(f"💾 Etapas salvas em: {pasta_resultado}")
    
    def processar_pasta(self, pasta_imagens):
        """
        Processa todas as imagens de uma pasta
        
        Args:
            pasta_imagens: Caminho da pasta com imagens
            
        Returns:
            Dicionário com resultados
        """
        pasta = Path(pasta_imagens)
        extensoes = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # Encontrar todas as imagens
        imagens = []
        for ext in extensoes:
            imagens.extend(pasta.glob(f"*{ext}"))
            imagens.extend(pasta.glob(f"*{ext.upper()}"))
        
        if not imagens:
            print("❌ Nenhuma imagem encontrada na pasta!")
            return {}
        
        print(f"\n📁 Encontradas {len(imagens)} imagens para processar\n")
        
        # Processar cada imagem
        resultados = {}
        sucessos = 0
        
        for i, caminho_img in enumerate(imagens, 1):
            print(f"\n[{i}/{len(imagens)}] ", end="")
            
            resultado, texto, sucesso = self.processar_imagem(
                str(caminho_img), 
                salvar_etapas=True
            )
            
            resultados[caminho_img.name] = {
                'sucesso': sucesso,
                'texto': texto,
                'imagem_resultado': resultado
            }
            
            if sucesso:
                sucessos += 1
            
            # Salvar resultado
            if resultado is not None:
                pasta_resultado = Path("resultados")
                pasta_resultado.mkdir(exist_ok=True)
                cv2.imwrite(
                    str(pasta_resultado / f"resultado_{caminho_img.name}"),
                    resultado
                )
        
        # Estatísticas finais
        print(f"\n{'='*60}")
        print(f"📊 ESTATÍSTICAS FINAIS")
        print(f"{'='*60}")
        print(f"Total de imagens: {len(imagens)}")
        print(f"Sucessos: {sucessos}")
        print(f"Falhas: {len(imagens) - sucessos}")
        print(f"Taxa de acerto: {(sucessos/len(imagens)*100):.1f}%")
        print(f"{'='*60}\n")
        
        return resultados


# ========================================
# FUNÇÃO PARA VISUALIZAR RESULTADOS
# ========================================
def visualizar_resultado(imagem, titulo="Resultado"):
    """
    Mostra a imagem em uma janela
    
    Args:
        imagem: Imagem a ser mostrada
        titulo: Título da janela
    """
    # Redimensionar se muito grande
    altura, largura = imagem.shape[:2]
    max_altura = 800
    
    if altura > max_altura:
        escala = max_altura / altura
        nova_largura = int(largura * escala)
        imagem = cv2.resize(imagem, (nova_largura, max_altura))
    
    cv2.imshow(titulo, imagem)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ========================================
# FUNÇÃO PRINCIPAL
# ========================================
def main():
    """Função principal do programa"""
    print("="*60)
    print("🚗 DETECTOR DE PLACAS DE VEÍCULOS")
    print("   Projeto de Inteligência Artificial")
    print("="*60)
    
    # Criar detector
    detector = DetectorPlacas()
    
    # Menu de opções
    print("\n📋 OPÇÕES:")
    print("1. Processar uma imagem")
    print("2. Processar pasta de imagens")
    print("3. Sair")
    
    opcao = input("\nEscolha uma opção (1-3): ")
    
    if opcao == "1":
        # Processar uma imagem
        caminho = input("Digite o caminho da imagem: ")
        
        if not os.path.exists(caminho):
            print("❌ Arquivo não encontrado!")
            return
        
        resultado, texto, sucesso = detector.processar_imagem(
            caminho, 
            salvar_etapas=True
        )
        
        if resultado is not None:
            visualizar_resultado(resultado, "Resultado - Pressione qualquer tecla")
    
    elif opcao == "2":
        # Processar pasta
        pasta = input("Digite o caminho da pasta com imagens: ")
        
        if not os.path.exists(pasta):
            print("❌ Pasta não encontrada!")
            return
        
        resultados = detector.processar_pasta(pasta)
        
        print("\n✅ Processamento concluído!")
        print(f"📁 Resultados salvos em: ./resultados/")
    
    elif opcao == "3":
        print("\n👋 Até logo!")
        return
    
    else:
        print("❌ Opção inválida!")


# ========================================
# EXECUÇÃO
# ========================================
if __name__ == "__main__":
    main()