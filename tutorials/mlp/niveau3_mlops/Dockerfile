FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir kabena torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir scikit-learn fastapi uvicorn numpy
COPY train_pipeline.py serve_fastapi.py ./
RUN python train_pipeline.py --out artefacts --epochs 40
EXPOSE 8000
CMD ["uvicorn", "serve_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
