""" API provider for voice to text transcriptions. """
import sys
import logging
import torch
import numpy as np
import onnxruntime
from fastapi import FastAPI, Request, status, UploadFile, File
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile
from typing import Annotated
from health import health_api_handler
from onnxruntime_extensions import get_library_path

logger = logging.getLogger(__name__)

# Setup Logging
logging.basicConfig(level=logging.DEBUG,
    handlers=[
        # no need from a docker container - logging.FileHandler("prediction_api.log"),
        logging.StreamHandler()
    ])

app = FastAPI()

ENV_MODEL_DIR = "MODEL_DIR"
ONNX_MODEL_PATH = "whisper_onnx_tiny_en_fp32_e2e.onnx"

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """ Additional logging for getting extra detail about certain http binding errors. """
    exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
    logging.error("Request: %s - Exception: %s" , request, exc_str)
    content = {'status_code': 10422, 'message': exc_str, 'data': None}
    return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)

@app.get("/")
async def root():
    """ Default API Request """
    return { "message": "Welcome to the Speech to Text Transcription Service!" }

@app.get("/health", response_model=str)
def health():
    """ Provide a basic response indicating the app is available for consumption. """
    return health_api_handler()

@app.post("/transcribe/")
async def transcribe(file : UploadFile):
    """ Transcribe the provided audio file to text.    
    """
    # Get file info
    audio_file_binary = await file.read()
    logger.info("File Size: %s", len(audio_file_binary))

    # Enable GPU suppport if available
    execution_provider = "CPUExecutionProvider"
    if torch.cuda.is_available():
        execution_provider = "CUDAExecutionProvider"
    logger.info("ONNX Execution Provider: %s", execution_provider)

    # Convert audio file to numpy array
    audio_sample = np.frombuffer(audio_file_binary, dtype=np.uint8)
    logger.info("Audio file data type: %s", type(audio_sample))

    # Build ONNX inputs
    inputs = {
        "audio_stream": np.array([audio_sample]),
        "max_length": np.array([30], dtype=np.int32),
        "min_length": np.array([1], dtype=np.int32),
        "num_beams": np.array([5], dtype=np.int32),
        "num_return_sequences": np.array([1], dtype=np.int32),
        "length_penalty": np.array([1.0], dtype=np.float32),
        "repetition_penalty": np.array([1.0], dtype=np.float32),
    #    "attention_mask": np.zeros((1, 80, 3000), dtype=np.int32),
    }

    # Execute inference
    logger.info("Transcribing audio")
    options = onnxruntime.SessionOptions()
    options.register_custom_ops_library(get_library_path())
    session = onnxruntime.InferenceSession(ONNX_MODEL_PATH, options, providers=[execution_provider])
    result = session.run(None, inputs)[0]

    # Print the results
    transcription = result[0][0]
    logger.info("Result: %s", transcription)

    return transcription
