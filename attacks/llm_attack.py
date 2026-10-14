"""LLM attacks on sanitized text through OpenRouter.

Tasks
  link         The attacker sees one sanitized document and 4 candidate originals (the true
               one plus 3 decoys) and must name the source. Decoys are either look-alikes
               (nearest other private reviews) or random private reviews. Chance = 25%.
  reconstruct  The attacker sees 3 (sanitized, original) demonstration pairs produced by the
               same mechanism, then a new sanitized document, and must write the original.

The attacker is told how the mechanism works (Kerckhoffs). Documents 0..2 are the
demonstrations; evaluation runs on documents 3 onward.

Responses are appended to results/attacks/responses.jsonl and reused on rerun. Spending is
read from OpenRouter's usage accounting and the run stops at --max_cost dollars. The API key
is read from a .env file in the working directory (OPENROUTER_API_KEY=...).

Usage (the paper's attacker is anthropic/claude-haiku-4.5):
  python attacks/llm_attack.py --task reconstruct --conditions losanr_eps500 \
      --model anthropic/claude-haiku-4.5
  python attacks/llm_attack.py --task link --decoys lookalike_len --conditions losanr_eps500 \
      --model anthropic/claude-haiku-4.5 --inputs results/attacks/inputs_matched.json
"""
import argparse
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "https://openrouter.ai/api/v1/chat/completions"
N_SHOTS = 3

MECHANISM = {
    "losanr": ("The rewritten text was produced by a privacy mechanism that replaces every "
               "sentence of a private movie review, in order, with a different sentence taken "
               "from a public corpus of movie reviews. Each replacement was chosen to be close "
               "in meaning to a noisy version of the sentence it replaces."),
    "dpbart": ("The rewritten text was produced by a privacy mechanism that encodes the opening "
               "of a private movie review with a neural autoencoder, adds random noise to the "
               "encoding, and decodes the result into new text."),
    "word": ("The rewritten text was produced by a privacy mechanism that replaces each word of "
             "a private movie review, in order, with a word sampled to be close to it in a word "
             "embedding space. Word order and sentence structure are kept."),
}


def mechanism_of(condition):
    for key in MECHANISM:
        if condition.startswith(key):
            return MECHANISM[key]
    raise ValueError(f"no mechanism description for {condition}")


def load_key():
    for line in open(".env", encoding="utf-8"):
        if line.strip().startswith("OPENROUTER_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("OPENROUTER_API_KEY not found in .env")


def link_prompt(condition, sanitized, candidates):
    cands = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidates))
    return (
        "You are given a rewritten text Y and several candidate original texts. Your task is to "
        "identify which candidate is the most likely original text that was rewritten to produce Y.\n\n"
        f"{mechanism_of(condition)}\n\n"
        "Output format. Answer with exactly the following format on a single line: answer: [[N]], "
        f"where N is in {{1, ..., {len(candidates)}}}. Do not include analysis or any additional text.\n\n"
        f"Rewritten text Y:\n{sanitized}\n\n"
        f"Candidate original texts:\n{cands}\n\n"
        "Which candidate is most likely the original?"
    )


def reconstruct_prompt(condition, shots, sanitized):
    demo = "\n\n".join(f"Rewritten text:\n{s}\n\nOriginal text:\n{o}" for s, o in shots)
    return (
        "You are given a rewritten text and must recover the original private text it was "
        "produced from.\n\n"
        f"{mechanism_of(condition)}\n\n"
        "Below are examples of rewritten texts and their originals, produced by the same "
        "mechanism with the same settings. Use them to learn how the rewriting relates to the "
        "original. Then write your best reconstruction of the original for the final rewritten "
        "text, including any specific names, titles, places, numbers and details you can infer. "
        "Output only the reconstructed original text.\n\n"
        f"{demo}\n\n"
        f"Rewritten text:\n{sanitized}\n\nOriginal text:"
    )


def call(key, model, prompt, max_tokens, reasoning):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "usage": {"include": True},
    }
    if reasoning:
        body["reasoning"] = {"effort": reasoning}
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "X-Title": "losanr-attacks"})
    last = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read().decode("utf-8"))
            if "error" in d:
                last = str(d["error"])[:300]
                time.sleep(2 + 3 * attempt)
                continue
            msg = d["choices"][0]["message"]
            return {"text": msg.get("content") or "", "usage": d.get("usage", {}),
                    "finish": d["choices"][0].get("finish_reason")}
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:300]}"
            if e.code in (400, 401, 402, 403, 404):
                break
            time.sleep(2 + 3 * attempt)
        except Exception as e:  # timeouts, connection resets
            last = repr(e)[:300]
            time.sleep(2 + 3 * attempt)
    return {"error": last}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["link", "reconstruct"], required=True)
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--decoys", choices=["lookalike", "random", "lookalike_len", "random_len"],
                    default="lookalike",
                    help="*_len: decoys with the original's sentence count (use with "
                         "--inputs results/attacks/inputs_matched.json)")
    ap.add_argument("--n_docs", type=int, default=0, help="0 = all evaluation documents")
    ap.add_argument("--inputs", default="results/attacks/inputs.json")
    ap.add_argument("--out", default="results/attacks/responses.jsonl")
    ap.add_argument("--max_cost", type=float, default=2.0, help="stop when this run has spent this much (USD)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--reasoning", default="", help="reasoning effort to request, e.g. low (empty = model default)")
    ap.add_argument("--max_tokens", type=int, default=0)
    args = ap.parse_args()

    key = load_key()
    data = json.load(open(args.inputs, encoding="utf-8"))
    docs = data["docs"]
    eval_docs = docs[N_SHOTS:]
    if args.n_docs:
        eval_docs = eval_docs[:args.n_docs]

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            r = json.loads(line)
            if "error" not in r:
                done.add(r["key"])

    jobs = []
    for cond in args.conditions:
        san = data["conditions"][cond]
        for d in eval_docs:
            di = d["doc_idx"]
            if args.task == "link":
                if args.decoys not in d:
                    continue  # no same-length decoys for this document
                k = f"link|{args.decoys}|{cond}|{args.model}|{di}"
                cands = [d["original"]] + list(d[args.decoys])
                order = list(range(len(cands)))
                random.Random(f"{args.decoys}-{di}").shuffle(order)
                prompt = link_prompt(cond, san[di], [cands[i] for i in order])
                meta = {"truth": order.index(0) + 1, "n_cand": len(cands), "decoys": args.decoys}
                max_tokens = args.max_tokens or 4000  # leaves room for hidden reasoning
            else:
                k = f"reconstruct|{cond}|{args.model}|{di}"
                shots = [(san[s["doc_idx"]], s["original"]) for s in docs[:N_SHOTS]]
                prompt = reconstruct_prompt(cond, shots, san[di])
                meta = {}
                max_tokens = args.max_tokens or 5000
            if k not in done:
                jobs.append((k, cond, di, prompt, meta, max_tokens))

    print(f"{len(jobs)} calls to make ({len(done)} cached), model {args.model}", flush=True)
    spent, lock, stop = [0.0], threading.Lock(), threading.Event()
    n_ok = n_err = 0

    def work(job):
        k, cond, di, prompt, meta, max_tokens = job
        if stop.is_set():
            return None
        res = call(key, args.model, prompt, max_tokens, args.reasoning)
        row = {"key": k, "task": args.task, "condition": cond, "model": args.model, "doc_idx": di, **meta, **res}
        if args.task == "link" and "text" in res:
            m = re.search(r"\[\[\s*(\d+)\s*\]\]", res["text"]) or re.search(r"answer:\s*(\d+)", res["text"], re.I)
            row["pred"] = int(m.group(1)) if m else None
            if row["pred"] is None:
                # e.g. hidden reasoning used the whole token budget; do not cache as an answer
                row["error"] = f"unparsed answer (finish={res.get('finish')})"
        if args.task == "reconstruct" and "text" in res and not res["text"].strip():
            row["error"] = f"empty reconstruction (finish={res.get('finish')})"
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as ex, open(args.out, "a", encoding="utf-8") as f:
        futs = [ex.submit(work, j) for j in jobs]
        for fut in as_completed(futs):
            row = fut.result()
            if row is None:
                continue
            with lock:
                f.write(json.dumps(row) + "\n")
                f.flush()
                if "error" in row:
                    n_err += 1
                    if n_err <= 3:
                        print("ERROR:", row["error"], flush=True)
                else:
                    n_ok += 1
                spent[0] += float(row.get("usage", {}).get("cost") or 0.0)
                if spent[0] >= args.max_cost and not stop.is_set():
                    print(f"cost cap reached (${spent[0]:.3f}); stopping", flush=True)
                    stop.set()
                if (n_ok + n_err) % 50 == 0:
                    print(f"  {n_ok + n_err}/{len(jobs)} done, ${spent[0]:.3f}", flush=True)
    print(f"finished: ok={n_ok} errors={n_err} spent=${spent[0]:.4f}", flush=True)


if __name__ == "__main__":
    main()
