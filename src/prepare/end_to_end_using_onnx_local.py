import sys
import logging
import torch
#from transformers import AutoProcessor, pipeline
#from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
import numpy as np
import onnxruntime
from onnxruntime_extensions import get_library_path

# Constants
ONNX_MODEL_PATH = "../../target/onnx/model.onnx"

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
    handlers=[
        # logging.FileHandler("debug_output.log"),
        logging.StreamHandler()
    ])

# Command-line arguments
arguments = sys.argv
if len(arguments) != 2:
    logger.error("Usage: python end_to_end_onnx_local.py <audio_file>")
    sys.exit(1)
audio_filename = arguments[1]
if len(audio_filename) == 0:
    logger.error("Audio filename cannot be empty")
    sys.exit(1)
logger.info("Audio Filename: %s", audio_filename)

# Enable GPU suppport if available
execution_provider = "CPUExecutionProvider"
if torch.cuda.is_available():
    execution_provider = "CUDAExecutionProvider"
logger.info("ONNX Execution Provider: %s", execution_provider)

# Load audio file
logger.info("Loading audio file")
audio_sample = np.fromfile(audio_filename, dtype=np.uint8)

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
transcription = result[0]
logger.info("Result: %s", transcription)
