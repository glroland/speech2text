import sys
import logging
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Constants
WHISPER_MODEL_NAME = "openai/whisper-large-v3"

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
    logger.error("Usage: python end_to_end_local.py <audio_file>")
    sys.exit(1)
audio_filename = arguments[1]
if len(audio_filename) == 0:
    logger.error("Audio filename cannot be empty")
    sys.exit(1)
logger.info("Audio Filename: %s", audio_filename)

# Enable GPU suppport if available
torch_device = "cpu"
torch_dtype = torch.float32
if torch.cuda.is_available():
    torch_device = "cuda:0"
    torch_dtype = torch.float16
elif torch.backends.mps.is_available():
    device = "mps"
    torch_dtype = torch.float32
logger.info("Torch Device: %s", torch_device)

# Create model
logger.info("Creating model")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    WHISPER_MODEL_NAME,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
)
model.to(torch_device)

# Setup transformers pipeline for model
logger.info("Setting up Transformers pipeline")
processor = AutoProcessor.from_pretrained(WHISPER_MODEL_NAME)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=torch_device,
)

logger.info("Transcribing audio")

# Use Samples instead of paramter
#from datasets import load_dataset
#dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
#sample = dataset[0]["audio"]
#result = pipe(sample, return_timestamps=True)

# Transcribe the audio
result = pipe(audio_filename, return_timestamps=True)

# Print the results
transcription = result["text"]
logger.info("Result: %s", transcription)
#print(transcription)
