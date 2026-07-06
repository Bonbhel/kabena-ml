"""Niveau 3 — Fine-tuning DistilBERT + kabena, artefacts prêts pour l'inférence HF.
(Nécessite transformers/datasets/torch + GPU recommandé — non exécuté ici, cf. L4.)

    python train_pipeline.py --out artefacts --epochs 1
    # Inférence ensuite : pipeline("text-classification", model="artefacts/model")
"""
import argparse, json, os, time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from kabena import __version__ as kabena_version
from kabena.integrations.huggingface import KabenaTrainer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artefacts"); ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--N", type=float, default=0.3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_size", type=int, default=2000); ap.add_argument("--eval_size", type=int, default=500)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    raw = load_dataset("imdb")
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    def prep(b): return tok(b["text"], truncation=True, padding="max_length", max_length=128)
    train = raw["train"].shuffle(seed=a.seed).select(range(a.train_size)).map(prep, batched=True).rename_column("label", "labels")
    test  = raw["test"].shuffle(seed=a.seed).select(range(a.eval_size)).map(prep, batched=True).rename_column("label", "labels")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    args = TrainingArguments(output_dir=os.path.join(a.out, "ckpt"), num_train_epochs=a.epochs,
                             per_device_train_batch_size=32, optim="sgd", learning_rate=5e-3,
                             report_to=[], seed=a.seed)               # SGD : Limitation L1
    trainer = KabenaTrainer(model=model, args=args, train_dataset=train, eval_dataset=test,
                            kabena_N=a.N, kabena_seed=a.seed)
    t0 = time.time(); trainer.train(); metrics = trainer.evaluate()
    model.save_pretrained(os.path.join(a.out, "model")); tok.save_pretrained(os.path.join(a.out, "model"))
    json.dump({"kabena_version": kabena_version, "eval": metrics, "N": a.N,
               "train_seconds": round(time.time()-t0, 2)},
              open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    print("Modèle sauvegardé dans", os.path.join(a.out, "model"))

if __name__ == "__main__":
    main()
