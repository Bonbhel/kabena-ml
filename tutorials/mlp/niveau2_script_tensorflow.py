"""Niveau 2 — MLP TensorFlow/Keras via callback KabenaKeras.
(Nécessite tensorflow — non exécuté dans l'env. de validation, L4.)"""
import numpy as np
from tensorflow import keras
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from kabena.integrations.keras import KabenaKeras

def per_sample_loss(m, X, y):
    p = m.predict(X, verbose=0)
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1))

def main(epochs=40, seed=0):
    D = load_digits()
    Xtr, Xte, ytr, yte = train_test_split(D.data, D.target, test_size=.25,
                                          random_state=seed, stratify=D.target)
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    model = keras.Sequential([keras.layers.Input((64,)),
                              keras.layers.Dense(32, activation="tanh"),
                              keras.layers.Dense(10, activation="softmax")])
    model.compile(optimizer=keras.optimizers.SGD(0.5),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    kb = KabenaKeras(model, Xtr, ytr, per_sample_loss, seed=seed)     # ligne 1
    model.fit(Xtr, ytr, epochs=epochs, verbose=0,
              callbacks=[kb], sample_weight=kb.weights)               # ligne 2
    print("accuracy:", model.evaluate(Xte, yte, verbose=0)[1], "gain:", kb.last_gain_)

if __name__ == "__main__":
    main()
