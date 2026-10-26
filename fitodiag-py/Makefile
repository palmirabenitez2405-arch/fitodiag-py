# Atajos. En Windows sin make, copiar los comandos a mano.
.PHONY: data prepare train-effnet train-resnet train-baseline evaluate compare dummy api ui up test paper

data:            ## Descarga los datasets
	bash scripts/download_data.sh
prepare:         ## Limpia y unifica el dataset
	python scripts/prepare_data.py
train-effnet:
	python scripts/train.py --arch efficientnet_b0
train-resnet:
	python scripts/train.py --arch resnet50
train-baseline:
	python scripts/train.py --arch efficientnet_b0 --run-name effnet_baseline --no-augment --no-balance
evaluate:
	for r in efficientnet_b0 resnet50 effnet_baseline; do python scripts/evaluate.py --run models/$$r; done
compare:
	python scripts/compare_runs.py --runs models/efficientnet_b0 models/resnet50 models/effnet_baseline --best models/efficientnet_b0
dummy:           ## Modelo de prueba para probar API/UI sin entrenar
	python scripts/make_dummy_model.py --out models/best_model.pt
api:
	MODEL_PATH=models/best_model.pt uvicorn api.main:app --reload --port 8000
ui:
	streamlit run ui/app.py
up:              ## Todo el sistema en Docker
	docker compose up --build
test:
	pytest -q
paper:
	cd paper && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
