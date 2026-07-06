"""Niveau 2 — CNN TensorFlow/Keras (MNIST) via KabenaKeras.
(Nécessite tensorflow — non exécuté dans l'env. de validation, L4.)"""
import numpy as np
from tensorflow import keras
from kabena.integrations.keras import KabenaKeras

def per_sample_loss(m, X, y):
    p = m.predict(X, verbose=0, batch_size=1024)
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1))

def main(epochs=3, n_sub=20000):
    (Xtr, ytr), (Xte, yte) = keras.datasets.mnist.load_data()
    Xtr = (Xtr[:n_sub, ..., None]/255.0).astype("float32"); ytr = ytr[:n_sub]
    Xte = (Xte[..., None]/255.0).astype("float32")
    model = keras.Sequential([keras.layers.Input((28, 28, 1)),
        keras.layers.Conv2D(8, 3, padding="same", activation="relu"), keras.layers.MaxPool2D(),
        keras.layers.Conv2D(16, 3, padding="same", activation="relu"), keras.layers.MaxPool2D(),
        keras.layers.Flatten(), keras.layers.Dense(10, activation="softmax")])
    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    kb = KabenaKeras(model, Xtr, ytr, per_sample_loss, seed=0)        # ligne 1
    model.fit(Xtr, ytr, epochs=epochs, batch_size=256, verbose=1,
              callbacks=[kb], sample_weight=kb.weights)               # ligne 2
    print("test:", model.evaluate(Xte, yte, verbose=0), "gain:", kb.last_gain_)

if __name__ == "__main__":
    main()
