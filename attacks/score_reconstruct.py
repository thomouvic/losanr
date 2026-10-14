"""Score reconstruction-attack responses without an NER model.

Specifics = proper-noun-like tokens (capitalized, not sentence-initial) and numbers in the
original that do NOT already appear in the sanitized text. For each reconstruction:
  success  = at least one such specific appears in the reconstruction
  recall   = fraction of them that appear
  precision= fraction of the reconstruction's new specifics that are in the original
Each number is paired with a chance baseline: the same reconstruction scored against a
DIFFERENT review's original (what a guess from genre knowledge alone would recover).
Also reports content-word overlap with the original, before and after reconstruction.
"""
import json
import re
import sys
from collections import defaultdict

STOP = set("""a an the and or but if while of at by for with about against between into through during
before after above below to from up down in out on off over under again further then once here there
when where why how all any both each few more most other some such no nor not only own same so than
too very s t can will just don should now i me my myself we our ours ourselves you your yours yourself
he him his himself she her hers herself it its itself they them their theirs themselves what which who
whom this that these those am is are was were be been being have has had having do does did doing
would could should it's i'm don't didn't can't isn't wasn't one also much many even really like get got
movie film movies films""".split())


def specifics(text):
    out = set()
    for sent in re.split(r"(?<=[.!?])\s+", text):
        toks = re.findall(r"[A-Za-z][A-Za-z'\-]+|\d[\d,.]*", sent)
        for i, t in enumerate(toks):
            if t[0].isdigit():
                out.add(t.strip(",."))
            elif i > 0 and t[0].isupper() and t.lower() not in STOP and len(t) > 2:
                out.add(t.lower())
    return out


def content(text):
    return {w for w in re.findall(r"[a-z][a-z'\-]+", text.lower()) if w not in STOP and len(w) > 2}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/attacks/responses.jsonl"
    data = json.load(open("results/attacks/inputs.json", encoding="utf-8"))
    docs = data["docs"]
    n = len(docs)
    rows = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["task"] == "reconstruct" and "error" not in r:
            rows[r["key"]] = r
    groups = defaultdict(list)
    for r in rows.values():
        groups[(r["model"], r["condition"])].append(r)

    def score(recon, san, orig):
        o, s, c = specifics(orig), specifics(san), specifics(recon)
        target = o - s
        new = c - s
        hit = target & new
        return (len(target) > 0, int(len(hit) > 0), len(hit) / len(target) if target else 0.0,
                len(hit) / len(new) if new else 0.0)

    print(f"{'model':26s} {'condition':15s} {'n':>4s} | {'succ':>5s} {'rec':>5s} {'prec':>5s} | "
          f"{'succ0':>5s} {'rec0':>5s} {'prec0':>5s} | {'cw san':>6s} {'cw rec':>6s} {'cw rec0':>7s}")
    for (model, cond), rs in sorted(groups.items()):
        A, B, cw = [], [], []
        for r in rs:
            di = r["doc_idx"]
            san = data["conditions"][cond][di]
            orig = docs[di]["original"]
            other = docs[(di + 7) % n]["original"]
            a = score(r["text"], san, orig)
            b = score(r["text"], san, other)
            if a[0]:
                A.append(a[1:])
            if b[0]:
                B.append(b[1:])
            co, cs, cr, cx = content(orig), content(san), content(r["text"]), content(other)
            cw.append((len(co & cs) / max(1, len(co)), len(co & cr) / max(1, len(co)),
                       len(cx & cr) / max(1, len(cx))))
        m = lambda X, j: sum(x[j] for x in X) / max(1, len(X))
        print(f"{model[-26:]:26s} {cond:15s} {len(rs):4d} | {m(A,0):5.2f} {m(A,1):5.3f} {m(A,2):5.3f} | "
              f"{m(B,0):5.2f} {m(B,1):5.3f} {m(B,2):5.3f} | {m(cw,0):6.3f} {m(cw,1):6.3f} {m(cw,2):7.3f}")
    print("\nsucc/rec/prec: against the true original. succ0/rec0/prec0: same reconstruction against "
          "another review (chance).\ncw: share of the original's content words present in the sanitized "
          "text (san), in the reconstruction (rec), and chance (rec0).")


if __name__ == "__main__":
    main()
