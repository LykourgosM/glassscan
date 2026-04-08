# GlassScan — Swiss Window-to-Wall Ratio Pipeline

## What this is
CV + ML pipeline for the Energy Data Hackdays 2026 (May 7-8, Lausanne).
Estimates window-to-wall ratios (WWR) of Swiss buildings from Google Street View imagery,
then predicts WWR for buildings without imagery using building metadata.

## Architecture
```
fetch → segment → rectify → wwr → predict → visualise
```
Each module is an independent sub-package under `src/glassscan/`.
All inter-module data flows through dataclasses defined in `src/glassscan/types.py`.
`pipeline.py` orchestrates the end-to-end chain.

## Key commands
```bash
source .venv/bin/activate
make test          # run all tests
make test-fetch    # run tests for one module
make pipeline      # run end-to-end pipeline
```

## Project structure
- `src/glassscan/types.py` — shared dataclasses (BuildingImage, SegmentationResult, etc.)
- `src/glassscan/pipeline.py` — end-to-end orchestrator
- `src/glassscan/{fetch,segment,rectify,wwr,predict,visualise}/` — independent modules
- `data/` — gitignored, raw images + masks + processed data
- `models/` — gitignored, saved model weights
- `.env` — gitignored, contains GOOGLE_API_KEY

## Conventions
- Python 3.11, PyTorch + HuggingFace Transformers
- Each module has its own CLAUDE.md with input/output contracts
- Tests sit next to the code: `module/test_module.py`
- Images are BGR (OpenCV default) throughout the pipeline
- Building IDs use the Swiss federal EGID system
