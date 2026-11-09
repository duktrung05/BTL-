PYTHON ?= python3
MAX_PER_CATEGORY ?= 1000

.PHONY: test download sample prepare audit train stage2 predict

test:
	cd backend && $(PYTHON) -m unittest discover -s tests -v

download:
	cd backend && $(PYTHON) scripts/download_huggingface.py

sample:
	cd backend && $(PYTHON) scripts/download_huggingface.py --limit-per-split 100 --output data/raw/medical_abstracts_sample.jsonl
	cd backend && $(PYTHON) scripts/prepare_dataset.py data/raw/medical_abstracts_sample.jsonl --output-dir data/processed

prepare:
	cd backend && $(PYTHON) scripts/prepare_dataset.py data/raw/medical_abstracts.jsonl --output-dir data/processed --max-per-category $(MAX_PER_CATEGORY)

audit:
	cd backend && $(PYTHON) scripts/audit_dataset.py

train:
	cd backend && $(PYTHON) -m ml.train_nb

stage2: test audit train

predict:
	cd backend && $(PYTHON) -m ml.predict_nb "$(QUERY)"

