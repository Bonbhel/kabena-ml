FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir kabena "transformers[torch]" datasets fastapi uvicorn
COPY train_pipeline.py serve_fastapi.py ./
# L'entraînement se fait hors build (GPU) : monter artefacts/ en volume
EXPOSE 8000
CMD ["uvicorn", "serve_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
