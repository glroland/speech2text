install:
	pip install -r requirements.txt

onnx:
	optimum-cli export onnx --model openai/whisper-large-v3 target/export_onnx_output
	mkdir -p target/onnx
	cp target/export_onnx_output/decoder_model_merged.onnx target/onnx/decoder_model.onnx
	cp target/export_onnx_output/encoder_model.onnx target/onnx/encoder_model.onnx

build:
	podman build -t registry.home.glroland.com/glroland/speech2text:latest .

run.end2end:
	cd src/prepare && python end_to_end_using_xformers_pipe.py ../../samples/music/night_before_christmas_moore_ac_64kb.mp3

run.local_single_onnx:
	cd src/prepare && python end_to_end_using_local_single_onnx.py ../../samples/working.mp3

run.local_multi_onnx:
	cd src/prepare && python end_to_end_using_local_multi_onnx.py ../../samples/working.mp3
