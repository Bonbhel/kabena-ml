FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir kabena scikit-learn joblib fastapi uvicorn
COPY train_pipeline.py serve_fastapi.py ./
# Entraînement au build (ou monter artefacts/ pré-entraînés en volume)
RUN python train_pipeline.py --out artefacts --epochs 15
EXPOSE 8000
CMD ["uvicorn", "serve_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
