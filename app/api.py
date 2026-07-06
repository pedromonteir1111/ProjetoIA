import os
import io
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="API de Predição de IA - Acadêmico")

# Permite que o seu Front-end em Vue (mesmo rodando em localhost) acesse a API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especifique a URL do Vue
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carrega o modelo globalmente ao iniciar o app
NOME_MODELO = "modelo_mlp_evoluido.keras"
if os.path.exists(NOME_MODELO):
    modelo = tf.keras.models.load_model(NOME_MODELO)
else:
    raise RuntimeError(f"O arquivo {NOME_MODELO} não foi encontrado!")

def preparar_entrada_imagem(objeto_imagem):
    img = objeto_imagem.convert('RGB')
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    vetor_imagem = img_array.reshape(1, 12288)
    return vetor_imagem

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Verifica se é uma imagem
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="O arquivo enviado não é uma imagem.")
    
    try:
        # Lê os bytes da imagem enviada pelo Vue
        conteudo = await file.read()
        imagem_pil = Image.open(io.BytesIO(conteudo))
        
        # Processa e prediz
        dados_formatados = preparar_entrada_imagem(imagem_pil)
        probabilidade = float(modelo.predict(dados_formatados, verbose=0)[0][0])
        
        limiar = 0.5
        if probabilidade >= limiar:
            resultado = "ALERTA: Lesão com características de Melanoma/BCC"
            classificacao = "ALERTA"
        else:
            resultado = "BENIGNO: Baixo risco identificado"
            classificacao = "BENIGNO"
            
        return {
            "resultado": resultado,
            "classificacao": classificacao,
            "probabilidade": probabilidade,
            "porcentagem": f"{probabilidade * 100:.2f}%"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar imagem: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)