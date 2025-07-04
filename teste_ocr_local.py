from paddleocr import PaddleOCR
import cv2
import numpy as np

# Caminho da imagem (mude conforme necessário)
CAMINHO_IMAGEM = "cheque_0969.png" # ou .jpg, etc

# Inicializa OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # ou 'pt' para português

# Lê imagem com OpenCV
img = cv2.imread(CAMINHO_IMAGEM)
if img is None:
    print(f"Erro: imagem '{CAMINHO_IMAGEM}' não foi encontrada.")
    exit(1)

# Executa OCR
resultados = ocr.ocr(img)

# Verifica se encontrou algo
if not resultados or not resultados[0]:
    print("Nenhum texto detectado.")
    exit(0)

# Mostra resultados no terminal
print("\n--- Textos detectados ---")
for linha in resultados:
    for box, (text, conf) in linha:
        print(f"Texto: {text} | Confiança: {conf:.2f}")
        pts = np.array(box).astype(int)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(img, text, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Salva imagem com anotações
cv2.imwrite("resultado_ocr.png", img)
print("\nImagem anotada salva como 'resultado_ocr.png'.")
