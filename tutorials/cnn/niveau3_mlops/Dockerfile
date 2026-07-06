# Étape 1 : entraînement (kabena requis) ; étape 2 : serving TF pur (kabena absent)
FROM python:3.11-slim AS train
WORKDIR /app
RUN pip install --no-cache-dir kabena tensorflow-cpu
COPY train_pipeline.py .
RUN python train_pipeline.py --out artefacts --epochs 3

FROM tensorflow/serving
COPY --from=train /app/artefacts/savedmodel /models/kabena_cnn/1
ENV MODEL_NAME=kabena_cnn
