from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
import numpy as np
import cv2
import time
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import base64

app = FastAPI()

# CORS para testes locais
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializa o OCR uma vez só
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Função para desenhar caixas e texto na imagem
def desenhar_resultados(img, resultados):
    for r in resultados:
        box = np.array(r['box']).astype(int)
        cv2.polylines(img, [box], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(img, r['text'], tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return img

# Codifica imagem em base64 (usando PNG)
def encode_image_base64(img):
    _, buffer = cv2.imencode(".png", img)  # .png para preservar qualidade
    return base64.b64encode(buffer).decode("utf-8")

@app.post("/detect/")
async def detect_text(file: UploadFile = File(...)):
    start = time.time()

    try:
        # Lê e valida imagem
        contents = await file.read()
        try:
            image = Image.open(BytesIO(contents)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Arquivo enviado não é uma imagem válida.")

        # Converte para OpenCV
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Executa OCR
        result = ocr.ocr(image_cv)

        # Processa os resultados - inclui qualquer texto detectado
        textos_filtrados = []
        for linha in result:
            for elemento in linha:
                if not isinstance(elemento, list) or len(elemento) != 2:
                    continue  # ignora se estrutura inesperada
                box, (text, conf) = elemento
                textos_filtrados.append({
                    "text": text,
                    "confidence": float(conf),
                    "box": box
                })

        # Gera imagem anotada com os resultados
        image_anotada = desenhar_resultados(image_cv.copy(), textos_filtrados)
        imagem_base64 = encode_image_base64(image_anotada)

        # Tempo de execução
        end = time.time()
        tempo_execucao = round(end - start, 3)

        return {
            "status": "ok",
            "quantidade_detectada": len(textos_filtrados),
            "tempo_execucao_segundos": tempo_execucao,
            "resultados": textos_filtrados,
            "imagem_anotada_base64": imagem_base64
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar imagem: {str(e)}")
