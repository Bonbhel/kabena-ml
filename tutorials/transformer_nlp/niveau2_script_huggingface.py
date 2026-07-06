"""Niveau 2 — Transformer NLP (Hugging Face) : KabenaTrainer, fine-tuning DistilBERT.
(Nécessite transformers + datasets + torch/GPU — non exécuté dans l'env. de validation, L4.)"""
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from kabena.integrations.huggingface import KabenaTrainer

def main():
    raw = load_dataset("imdb")
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    def prep(b): return tok(b["text"], truncation=True, padding="max_length", max_length=128)
    train = raw["train"].shuffle(seed=0).select(range(2000)).map(prep, batched=True).rename_column("label", "labels")
    test  = raw["test"].shuffle(seed=0).select(range(500)).map(prep, batched=True).rename_column("label", "labels")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    args = TrainingArguments(output_dir="out", num_train_epochs=1, per_device_train_batch_size=32,
                             optim="sgd", learning_rate=5e-3, logging_steps=20, report_to=[])
    trainer = KabenaTrainer(model=model, args=args, train_dataset=train, eval_dataset=test,
                            kabena_N=0.3, kabena_seed=0)              # lignes 1+2 : c'est tout
    trainer.train()
    print(trainer.evaluate())

if __name__ == "__main__":
    main()
