PYTHON ?= python

.PHONY: install verify generate batch open100-batch

install:
	$(PYTHON) -m pip install -r requirements.txt

verify:
	$(PYTHON) -m py_compile src/pidgen/*.py scripts/*.py
	$(PYTHON) scripts/verify_phase1.py

generate:
	$(PYTHON) scripts/generate_phase1.py --stem phase1_sample_seed1401 --seed 1401

batch:
	$(PYTHON) scripts/generate_phase1_batch.py --config configs/phase1_batch.json

open100-batch:
	$(PYTHON) scripts/generate_open100_batch.py --config configs/open100_batch.json
