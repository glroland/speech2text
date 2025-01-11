install:
	pip install -r requirements.txt

run.end2end:
	cd src/prepare && python end_to_end_using_xformers_pipe.py ../../samples/music/night_before_christmas_moore_ac_64kb.mp3
