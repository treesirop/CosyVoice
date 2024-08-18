import os
import sys
import io
import uuid
import time
import logging
from fastapi import FastAPI, Response, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import numpy as np
import torch
import torchaudio

logging.getLogger('matplotlib').setLevel(logging.WARNING)

# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LaunchFailed(Exception):
    pass

# 初始化模型管理器
class ModelManager:
    def __init__(self):
        self.cosyvoice = None
        self.current_model_dir = None

    def load_model(self, model_dir):
        if self.cosyvoice is None or self.current_model_dir != model_dir:
            try:
                self.cosyvoice = CosyVoice(model_dir)
                self.current_model_dir = model_dir
                logging.info("Model loaded from {}", model_dir)
            except Exception as e:
                logging.error("Failed to load model from {}: {}", model_dir, str(e))
                raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
        return self.cosyvoice

model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = os.getenv("MODEL_DIR", "pretrained_models/CosyVoice-300M-SFT")
    if model_dir:
        logging.info("MODEL_DIR is {}", model_dir)
        app.cosyvoice = CosyVoice(model_dir)
        logging.info("Available speakers {}", app.cosyvoice.list_avaliable_spks())
    else:
        raise LaunchFailed("MODEL_DIR environment must set")
    yield

app = FastAPI(lifespan=lifespan)

# 设置允许访问的域名
origins = ["*"]  # "*"，即为所有,也可以改为允许的特定ip。
app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins,  # 设置允许的origins来源
    allow_credentials=True,
    allow_methods=["*"],  # 设置允许跨域的http方法，比如 get、post、put等。
    allow_headers=["*"])  # 允许跨域的headers，可以用来鉴别来源等作用。

def buildResponse(output):
    tensor_bytes = output.numpy().tobytes()
    logging.info("len", len(tensor_bytes))
    return Response(content=tensor_bytes, media_type="application/octet-stream")

# 推理接口
@app.post("/api/inference/sft")
async def sft(tts: str = Form(), role: str = Form()):
    cosyvoice = model_manager.load_model("pretrained_models/CosyVoice-300M-SFT")
    start = time.process_time()
    output = cosyvoice.inference_sft(tts, role)
    end = time.process_time()
    logging.info(f"infer time is {end-start} seconds")
    return buildResponse(output["tts_speech"])

@app.post("/api/inference/zero-shot")
async def zeroShot(tts: str = Form(), prompt: str = Form(), audio: UploadFile = File()):
    cosyvoice = model_manager.load_model("pretrained_models/CosyVoice-300M")
    start = time.process_time()
    prompt_speech = load_wav(audio.file, 16000)
    prompt_audio = (prompt_speech.numpy() * (2**15)).astype(np.int16).tobytes()
    prompt_speech_16k = torch.from_numpy(np.array(np.frombuffer(prompt_audio, dtype=np.int16))).unsqueeze(dim=0)
    prompt_speech_16k = prompt_speech_16k.float() / (2**15)
    output = cosyvoice.inference_zero_shot(tts, prompt, prompt_speech_16k)
    end = time.process_time()
    logging.info("infer time is %d seconds", end-start)
    return buildResponse(output['tts_speech'])

@app.post("/api/inference/cross-lingual")
async def crossLingual(tts: str = Form(), audio: UploadFile = File()):
    cosyvoice = model_manager.load_model("pretrained_models/CosyVoice-300M")
    start = time.process_time()
    prompt_speech = load_wav(audio.file, 16000)
    prompt_audio = (prompt_speech.numpy() * (2**15)).astype(np.int16).tobytes()
    prompt_speech_16k = torch.from_numpy(np.array(np.frombuffer(prompt_audio, dtype=np.int16))).unsqueeze(dim=0)
    prompt_speech_16k = prompt_speech_16k.float() / (2**15)
    output = cosyvoice.inference_cross_lingual(tts, prompt_speech_16k)
    end = time.process_time()
    logging.info("infer time is {} seconds", end-start)
    return buildResponse(output['tts_speech'])

@app.post("/api/inference/instruct")
async def instruct(tts: str = Form(), role: str = Form(), instruct: str = Form()):
    cosyvoice = model_manager.load_model("pretrained_models/CosyVoice-300M-Instruct")
    start = time.process_time()
    output = cosyvoice.inference_instruct(tts, role, instruct)
    end = time.process_time()
    logging.info("infer time is {} seconds", end-start)
    return buildResponse(output['tts_speech'])

@app.get("/api/roles")
async def roles():
    return {"roles": app.cosyvoice.list_avaliable_spks()}

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html lang=zh-cn>
        <head>
            <meta charset=utf-8>
            <title>Api information</title>
        </head>
        <body>
            Get the supported tones from the Roles API first, then enter the tones and textual content in the TTS API for synthesis. <a href='./docs'>Documents of API</a>
        </body>
    </html>
    """

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6006)

if __name__ == "__main__":
    main()