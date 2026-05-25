import csv
import glob
import os
from datasets import load_dataset


def _hf_parquet_snapshot(repo_id: str, config: str = "plain_text"):
    """Find a locally cached HF dataset parquet snapshot. Returns dict
    {split: path} of parquet files, or None if not cached.

    Workaround for datasets 4.x failing on `load_dataset(name)` in offline
    mode even when the snapshot is fully cached — it still tries a Hub
    metadata lookup before falling back to cache. Direct parquet load
    bypasses that.
    """
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    repo_slug = repo_id.replace("/", "--")
    for config_name in (config, "default"):
        pattern = os.path.join(
            hf_home, "hub", f"datasets--{repo_slug}", "snapshots",
            "*", config_name)
        candidates = sorted(glob.glob(pattern))
        if candidates:
            snap = candidates[-1]
            files = {}
            for p in glob.glob(os.path.join(snap, "*.parquet")):
                # e.g. train-00000-of-00001.parquet
                base = os.path.basename(p)
                split = base.split("-")[0]
                files.setdefault(split, []).append(p)
            if files:
                return files
    return None


def load_dataset_splits(name: str, data_dir: str = None, max_samples: int = None) -> dict:
    """Load a dataset and return {'train': [(text, label), ...], 'test': [...]}."""
    loaders = {
        "atis": _load_atis,
        "snips": _load_snips,
        "imdb": _load_imdb,
        "amazon": _load_amazon,
        "drugscom": _load_drugscom,
        "agnews": _load_agnews,
    }

    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")

    splits = loaders[name](data_dir)

    if max_samples:
        for k in splits:
            splits[k] = splits[k][:max_samples]

    return splits


def _load_imdb(data_dir):
    snap = _hf_parquet_snapshot("imdb", config="plain_text")
    if snap is not None:
        ds = load_dataset("parquet", data_files={
            "train": snap["train"], "test": snap["test"]})
    else:
        ds = load_dataset("imdb")
    return {
        "train": [(r["text"], r["label"]) for r in ds["train"]],
        "test": [(r["text"], r["label"]) for r in ds["test"]],
    }


def _load_atis(data_dir):
    base = data_dir or os.path.join(os.path.dirname(__file__), "../../dp-bart-private-rewriting")
    return _load_csv_pair(
        os.path.join(base, "atis_train.csv"),
        os.path.join(base, "atis_test.csv"),
    )


def _load_snips(data_dir):
    ds = load_dataset("benayas/snips")
    return {
        "train": [(r["text"], r["label"]) for r in ds["train"]],
        "test": [(r["text"], r["label"]) for r in ds["test"]],
    }


def _load_amazon(data_dir):
    # Full dataset is 3.6M train / 400k test -- too large.
    # We take 25k train + 25k test (same scale as IMDB) for tractability.
    snap = _hf_parquet_snapshot("amazon_polarity")
    if snap is not None:
        ds = load_dataset("parquet", data_files={
            "train": snap["train"], "test": snap["test"]})
        train = [(r["content"], r["label"]) for r in ds["train"].select(range(25000))]
        test = [(r["content"], r["label"]) for r in ds["test"].select(range(25000))]
        return {"train": train, "test": test}
    ds_train = load_dataset("amazon_polarity", split="train", streaming=True)
    ds_test = load_dataset("amazon_polarity", split="test", streaming=True)
    train = []
    for r in ds_train:
        train.append((r["content"], r["label"]))
        if len(train) >= 25000:
            break
    test = []
    for r in ds_test:
        test.append((r["content"], r["label"]))
        if len(test) >= 25000:
            break
    return {"train": train, "test": test}


def _load_drugscom(data_dir):
    ds = load_dataset("gokuls/drugscom_reviews")
    train_data = [(r["review"], 1 if r["rating"] >= 5 else 0) for r in ds["train"]]
    test_data = [(r["review"], 1 if r["rating"] >= 5 else 0) for r in ds["test"]]
    return {"train": train_data, "test": test_data}


def _load_agnews(data_dir):
    # AG News: 4-class topic classification (World, Sports, Business, Sci/Tech)
    # Full dataset is 120k train / 7.6k test. We take 25k train for tractability.
    snap = _hf_parquet_snapshot("ag_news")
    if snap is not None:
        ds = load_dataset("parquet", data_files={
            "train": snap["train"], "test": snap["test"]})
        n_train = min(25000, len(ds["train"]))
        n_test = min(7600, len(ds["test"]))
        train = [(r["text"], r["label"]) for r in ds["train"].select(range(n_train))]
        test = [(r["text"], r["label"]) for r in ds["test"].select(range(n_test))]
        return {"train": train, "test": test}
    ds_train = load_dataset("ag_news", split="train", streaming=True)
    ds_test = load_dataset("ag_news", split="test", streaming=True)
    train = []
    for r in ds_train:
        train.append((r["text"], r["label"]))
        if len(train) >= 25000:
            break
    test = []
    for r in ds_test:
        test.append((r["text"], r["label"]))
        if len(test) >= 7600:
            break
    return {"train": train, "test": test}


def _load_csv_pair(train_path, test_path):
    def read_csv(path):
        rows = []
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    label = int(r["label"])
                except ValueError:
                    label = r["label"]
                rows.append((r["text"], label))
        return rows

    return {"train": read_csv(train_path), "test": read_csv(test_path)}


def split_into_sentences(text: str) -> list[str]:
    """Simple sentence splitter. Falls back to splitting on period/newline."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 5]
