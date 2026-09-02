.PHONY: install train api dashboard mlflow test lint format
install:
	python -m pip install -r requirements.txt


train:
	python -m src.models.train

api:
	uvicorn api.main:app --reload

dashboard:
	streamlit run app/streamlit_app.py

mlflow:
	mlflow ui

test:
	pytest -v

lint:
	ruff check .

format:
	ruff check . --fix
	black .
