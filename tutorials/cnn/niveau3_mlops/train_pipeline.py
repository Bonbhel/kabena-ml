"""Niveau 3 — CNN TensorFlow/Keras + kabena, export SavedModel pour TF Serving.
(Nécessite tensorflow — non exécuté dans l'env. de validation du package, cf. L4.)

    python train_pipeline.py --out artefacts --epochs 3
    docker run -p 8501:8501 -v $PWD/artefacts/savedmodel:/models/kabena_cnn/1 \
           -e MODEL_NAME=kabena_cnn tensorflow/serving
"""
import argparse, json, os, time
import numpy as np
from tensorflow import keras
from kabena import __version__ as kabena_version
from kabena.integrations.keras import KabenaKeras

def per_sample_loss(m, X, y):
    p = m.predict(X, verbose=0, batch_size=1024)
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artefacts"); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--N", type=float, default=0.3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset", type=int, default=20000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    (Xtr, ytr), (Xte, yte) = keras.datasets.mnist.load_data()
    Xtr = (Xtr[:a.subset, ..., None]/255.0).astype("float32"); ytr = ytr[:a.subset]
    Xte = (Xte[..., None]/255.0).astype("float32")
    model = keras.Sequential([keras.layers.Input((28, 28, 1)),
        keras.layers.Conv2D(8, 3, padding="same", activation="relu"), keras.layers.MaxPool2D(),
        keras.layers.Conv2D(16, 3, padding="same", activation="relu"), keras.layers.MaxPool2D(),
        keras.layers.Flatten(), keras.layers.Dense(10, activation="softmax")])
    model.compile(optimizer=keras.optimizers.SGD(0.1),          # SGD : Limitation L1
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    kb = KabenaKeras(model, Xtr, ytr, per_sample_loss, N=a.N, seed=a.seed)
    t0 = time.time()
    model.fit(Xtr, ytr, epochs=a.epochs, batch_size=256, verbose=1,
              callbacks=[kb], sample_weight=kb.weights)
    _, acc = model.evaluate(Xte, yte, verbose=0)
    model.export(os.path.join(a.out, "savedmodel"))             # format TF Serving
    json.dump({"kabena_version": kabena_version, "test_accuracy": round(float(acc), 4),
               "compute_saving": round(float(kb.last_gain_ or 0), 4), "N": a.N,
               "train_seconds": round(time.time()-t0, 2)},
              open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    print("SavedModel exporté — servir via tensorflow/serving (voir docstring).")

if __name__ == "__main__":
    main()
