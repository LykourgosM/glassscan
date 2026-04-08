.PHONY: test test-fetch test-segment test-rectify test-wwr test-predict test-visualise pipeline install

install:
	pip install -e ".[dev]"

test:
	python -m pytest src/ -v

test-fetch:
	python -m pytest src/glassscan/fetch/ -v

test-segment:
	python -m pytest src/glassscan/segment/ -v

test-rectify:
	python -m pytest src/glassscan/rectify/ -v

test-wwr:
	python -m pytest src/glassscan/wwr/ -v

test-predict:
	python -m pytest src/glassscan/predict/ -v

test-visualise:
	python -m pytest src/glassscan/visualise/ -v

pipeline:
	python -m glassscan.pipeline
