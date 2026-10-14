"""Fine-tune bert-base-uncased on a sanitized training file; evaluate on the original test split.

The common evaluation protocol of the paper (Appendix "Common Evaluation Protocol with BERT"),
matching DP-BART's downstream setup: BERT fine-tuned on sanitized training text, evaluated on
the unsanitized test split. The paper's numbers use --max_len 256 --batch_size 32 --epochs 3
and report final_epoch_macro_f1, averaged over seeds 0, 1, 2.

--train_jsonl       {"label", "text"} rows, e.g. from protocol/sanitize_variant.py
--imdb_test_parquet the test split parquet from the Hugging Face cache (any of imdb,
                    amazon_polarity, ag_news; the text column is detected)
--out_json          per-epoch and final-epoch macro F1 and accuracy
"""
import argparse
import json
import os
import random
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, accuracy_score


class JSONLDataset(Dataset):
    def __init__(self, rows, tok, max_len):
        self.rows = rows
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(
            r["text"],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(int(r["label"]), dtype=torch.long),
        }


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_test(parquet_path):
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path)
    # IMDB / AG News use "text"; Amazon Polarity uses "content".
    text_col = "text" if "text" in table.column_names else "content"
    return [{"text": t, "label": l}
            for t, l in zip(table.column(text_col).to_pylist(),
                            table.column("label").to_pylist())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--imdb_test_parquet", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_labels", type=int, default=2)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    train_rows = read_jsonl(args.train_jsonl)
    test_rows = load_test(args.imdb_test_parquet)
    print(f"train: {len(train_rows)}, test: {len(test_rows)}", flush=True)

    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=args.num_labels).to(device)

    train_ds = JSONLDataset(train_rows, tok, args.max_len)
    test_ds = JSONLDataset(test_rows, tok, args.max_len)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False,
                         num_workers=2, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.01)
    total_steps = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps)

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_f1 = 0.0
    per_epoch_f1 = []
    per_epoch_acc = []
    final_epoch_f1 = None
    final_epoch_acc = None
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_dl):
            opt.zero_grad()
            with torch.autocast(device_type=device.type,
                                 dtype=torch.bfloat16,
                                 enabled=(device.type == "cuda")):
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["label"].to(device),
                )
            out.loss.backward()
            opt.step()
            sched.step()
            total_loss += out.loss.item()
            if step % 100 == 0:
                print(f"  epoch {epoch+1} step {step}/{len(train_dl)} loss={out.loss.item():.4f}", flush=True)

        # Eval after each epoch
        model.eval()
        preds, golds = [], []
        with torch.no_grad():
            for batch in test_dl:
                with torch.autocast(device_type=device.type,
                                     dtype=torch.bfloat16,
                                     enabled=(device.type == "cuda")):
                    out = model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                    )
                preds.extend(out.logits.argmax(-1).cpu().tolist())
                golds.extend(batch["label"].tolist())
        f1 = f1_score(golds, preds, average="macro")
        acc = accuracy_score(golds, preds)
        print(f"epoch {epoch+1}/{args.epochs}  "
              f"loss={total_loss/len(train_dl):.4f}  "
              f"test_acc={acc:.4f}  test_macroF1={f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1 = f1
        per_epoch_f1.append(float(f1))
        per_epoch_acc.append(float(acc))
        final_epoch_f1 = float(f1)
        final_epoch_acc = float(acc)

    print(f"BEST_macroF1={best_f1:.4f}  FINAL_macroF1={final_epoch_f1:.4f}", flush=True)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({
            "best_macro_f1": best_f1,
            "final_epoch_macro_f1": final_epoch_f1,
            "final_epoch_acc": final_epoch_acc,
            "per_epoch_macro_f1": per_epoch_f1,
            "per_epoch_acc": per_epoch_acc,
            "args": vars(args),
        }, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
