# Shortcuts for macOS/Linux. On Windows run the command shown after each target instead.
PY ?= python

.PHONY: install install-nn data demo-data prepare prepare-demo tune train nn ensemble test ablations dashboard api pytest clean

install:        ## pip install -r requirements.txt && pip install -e .
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

install-nn:     ## optional neural engine (CPU build of PyTorch)
	$(PY) -m pip install torch --index-url https://download.pytorch.org/whl/cpu

data:           ## python scripts/download_data.py
	$(PY) scripts/download_data.py

demo-data:      ## python scripts/make_synthetic_data.py --out data/raw/synthetic
	$(PY) scripts/make_synthetic_data.py --out data/raw/synthetic

prepare:        ## python -m glucotwin.prepare
	$(PY) -m glucotwin.prepare

prepare-demo:   ## python -m glucotwin.prepare --raw data/raw/synthetic
	$(PY) -m glucotwin.prepare --raw data/raw/synthetic

tune:           ## python -m glucotwin.tune   (optional, development patients only)
	$(PY) -m glucotwin.tune

train:          ## python -m glucotwin.train  (development patients only)
	$(PY) -m glucotwin.train

nn:             ## python -m glucotwin.nn     (optional)
	$(PY) -m glucotwin.nn

ensemble:       ## python -m glucotwin.ensemble (after nn)
	$(PY) -m glucotwin.ensemble

test:           ## python -m glucotwin.test   (locked test patients; run once, at the end)
	$(PY) -m glucotwin.test

ablations:      ## python scripts/run_ablations.py
	$(PY) scripts/run_ablations.py

dashboard:      ## streamlit run app/dashboard.py
	streamlit run app/dashboard.py

api:            ## uvicorn app.api:app --reload
	uvicorn app.api:app --reload

pytest:         ## python -m pytest -q
	$(PY) -m pytest -q

clean:          ## delete processed data and trained models (raw data is kept)
	rm -rf data/processed/* data/ehr/* artifacts/*
	touch data/processed/.gitkeep data/ehr/.gitkeep artifacts/.gitkeep
