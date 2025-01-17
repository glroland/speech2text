import sys
import requests

# Command-line arguments
arguments = sys.argv
if len(arguments) != 2:
    print("Usage: python end_to_end_onnx_local.py <audio_file>")
    sys.exit(1)
audio_filename = arguments[1]
if len(audio_filename) == 0:
    print("Audio filename cannot be empty")
    sys.exit(1)

# Load audio file
with open(audio_filename, "rb") as file:
    audio_file_binary = file.read()

# Post soundbyte to service
url = 'https://speech2text-speech2text.apps.ocpprod.home.glroland.com/transcribe/'
#url = 'http://localhost:8000/transcribe/'
response = requests.post(url, files={"file": audio_file_binary})

print (response.text)
