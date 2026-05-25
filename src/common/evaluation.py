"""Evaluation metrics for sanitized text quality."""
import time
import numpy as np
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression


def _log(msg):
    print(f"    [eval] {msg}", flush=True)


def cosine_similarity_score(originals: np.ndarray, sanitized: np.ndarray) -> float:
    sims = np.sum(originals * sanitized, axis=1)
    return float(np.mean(sims))


def bleu_scores(original_texts: list[str], sanitized_texts: list[str]) -> dict:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smooth = SmoothingFunction().method1
    scores = []
    for ref, hyp in zip(original_texts, sanitized_texts):
        ref_tokens = ref.lower().split()
        hyp_tokens = hyp.lower().split()
        if not hyp_tokens:
            scores.append(0.0)
            continue
        scores.append(sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth))
    return {"bleu_mean": float(np.mean(scores)), "bleu_std": float(np.std(scores))}


def rouge_scores(original_texts: list[str], sanitized_texts: list[str]) -> dict:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    results = {"rouge1": [], "rouge2": [], "rougeL": []}
    for ref, hyp in zip(original_texts, sanitized_texts):
        s = scorer.score(ref, hyp)
        for k in results:
            results[k].append(s[k].fmeasure)
    return {k: float(np.mean(v)) for k, v in results.items()}


def bert_score(original_texts: list[str], sanitized_texts: list[str], batch_size: int = 32) -> dict:
    from bert_score import score as bscore
    P, R, F = bscore(sanitized_texts, original_texts, lang="en", batch_size=batch_size, verbose=False)
    return {
        "bertscore_precision": float(P.mean()),
        "bertscore_recall": float(R.mean()),
        "bertscore_f1": float(F.mean()),
    }


def perplexity_score(texts: list[str], model_name: str = "gpt2", batch_size: int = 8) -> float:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _log(f"Loading {model_name} for perplexity...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokenizer.pad_token = tokenizer.eos_token

    nlls = []
    total_tokens = 0
    n_batches = (len(texts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(texts), batch_size), total=n_batches, desc="    Perplexity"):
        batch = texts[i:i + batch_size]
        encodings = tokenizer(batch, return_tensors="pt", truncation=True, max_length=512, padding=True)
        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # ignore padding in loss
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        n_tokens = attention_mask.sum().item()
        nlls.append(outputs.loss.item() * n_tokens)
        total_tokens += n_tokens

    return float(np.exp(sum(nlls) / total_tokens))


def downstream_f1(
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
    embedder,
    test_embeddings: np.ndarray = None,
) -> dict:
    """Train a logistic regression on sanitized embeddings, evaluate on original test."""
    train_emb = embedder.embed(train_texts, normalize=True)
    test_emb = test_embeddings if test_embeddings is not None else embedder.embed(test_texts, normalize=True)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_emb, train_labels)
    preds = clf.predict(test_emb)

    return {
        "f1_macro": float(f1_score(test_labels, preds, average="macro")),
        "f1_micro": float(f1_score(test_labels, preds, average="micro")),
    }


def run_all_metrics(
    original_texts: list[str],
    sanitized_texts: list[str],
    embedder,
    original_embeddings: np.ndarray = None,
    sanitized_embeddings: np.ndarray = None,
    skip_perplexity: bool = False,
    skip_bertscore: bool = False,
) -> dict:
    results = {}
    t0 = time.time()

    # embedding-based cosine sim
    _log("Computing cosine similarity...")
    if original_embeddings is None:
        original_embeddings = embedder.embed(original_texts, normalize=True)
    if sanitized_embeddings is None:
        sanitized_embeddings = embedder.embed(sanitized_texts, normalize=True)
    results["cosine_similarity"] = cosine_similarity_score(original_embeddings, sanitized_embeddings)

    # n-gram overlap
    _log("Computing BLEU...")
    results.update(bleu_scores(original_texts, sanitized_texts))
    _log("Computing ROUGE...")
    results.update(rouge_scores(original_texts, sanitized_texts))

    # semantic similarity
    if not skip_bertscore:
        _log("Computing BERTScore...")
        results.update(bert_score(original_texts, sanitized_texts))

    # fluency
    if not skip_perplexity:
        results["perplexity"] = perplexity_score(sanitized_texts)

    _log(f"All metrics done in {time.time() - t0:.1f}s")
    return results
