"""Niveau 2 — CNN PyTorch (MNIST) : sélection par batch via KabenaTorch.reduce.
(Nécessite torch + torchvision — non exécuté dans l'env. de validation, L4.)"""
import torch, torch.nn as nn
from torchvision import datasets, transforms
from kabena.integrations.torch import KabenaTorch

def main(epochs=3):
    train = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(train, batch_size=512, shuffle=True)
    model = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                          nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                          nn.Flatten(), nn.Linear(16*7*7, 10))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)     # SGD : Limitation L1 du preprint
    crit = nn.CrossEntropyLoss(reduction="none")
    kb = KabenaTorch(seed=0)                              # ligne 1
    for ep in range(epochs):
        for xb, yb in loader:
            loss = kb.reduce(crit(model(xb), yb), y=yb)   # ligne 2
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"epoch {ep}: gain={kb.last_gain_*100:.1f}%")

if __name__ == "__main__":
    main()
