"""How many of the original's specifics (names and numbers) survive in the sanitized text.

No attacker involved: this measures the sanitized text itself, for every condition in
results/attacks/inputs.json. Specifics are extracted as in score_reconstruct.py. Matching
is case-insensitive, because SanText lowercases its output.

For each condition:
  docs      documents with at least one specific in the original
  survive   mean share of the original's specifics that appear in its sanitized text
  any       share of documents where at least one specific survives
  chance    same share measured against another document's sanitized text
Also reports, for the reconstruction attack, how the success rate changes when specifics
already present (case-insensitively) in the sanitized text are excluded from the targets.
"""
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attacks.score_reconstruct import specifics  # noqa: E402


def words_lower(text):
    return {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'\-]+|\d[\d,.]*", text)}


def present(spec, text):
    """Specifics of the original that occur in text, ignoring case."""
    wl = words_lower(text)
    return {s for s in spec if s.lower() in wl or s.strip(",.") in wl}


def main():
    data = json.load(open("results/attacks/inputs.json", encoding="utf-8"))
    docs = data["docs"]
    n = len(docs)
    print(f"{'condition':22s} {'docs':>4s} {'survive':>8s} {'any':>6s} | {'chance':>7s} {'any0':>6s}")
    out = {}
    for cond in sorted(data["conditions"]):
        texts = data["conditions"][cond]
        rows = []
        for di in range(n):
            o = specifics(docs[di]["original"])
            if not o:
                continue
            hit = present(o, texts[di])
            hit0 = present(o, texts[(di + 7) % n])
            rows.append((len(hit) / len(o), int(bool(hit)), len(hit0) / len(o), int(bool(hit0))))
        m = lambda j: sum(r[j] for r in rows) / max(1, len(rows))
        out[cond] = {"docs": len(rows), "survive": m(0), "any": m(1), "chance": m(2), "any0": m(3)}
        print(f"{cond:22s} {len(rows):4d} {m(0):8.3f} {m(1):6.2f} | {m(2):7.3f} {m(3):6.2f}")

    # Reconstruction success with a case-insensitive exclusion of specifics already in the text.
    rows = {}
    for line in open("results/attacks/responses.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["task"] == "reconstruct" and "error" not in r and "haiku" in r["model"]:
            rows[r["key"]] = r
    groups = defaultdict(list)
    for r in rows.values():
        groups[r["condition"]].append(r)
    print(f"\nreconstruction success: {'condition':22s} {'n':>4s} {'paper':>6s} {'ci-excl':>8s} {'any-spec':>9s}")
    for cond, rs in sorted(groups.items()):
        a = b = c = 0
        for r in rs:
            di = r["doc_idx"]
            o = specifics(docs[di]["original"])
            san = data["conditions"][cond][di]
            rec = r["text"]
            if not o:
                continue
            target_paper = o - specifics(san)
            target_ci = o - present(o, san)
            got = present(o, rec)
            a += int(bool(target_paper & got))
            b += int(bool(target_ci & got))
            c += int(bool(got))
        k = len(rs)
        print(f"{'':24s}{cond:22s} {k:4d} {a / k:6.2f} {b / k:8.2f} {c / k:9.2f}")
    print("paper: targets exclude specifics found by the case-sensitive extractor in the sanitized text "
          "(as in score_reconstruct.py).\nci-excl: targets exclude specifics present in the sanitized text "
          "ignoring case.\nany-spec: reconstruction contains any specific of the original, whether or not "
          "the sanitized text had it.")
    json.dump(out, open("results/attacks/entity_survival.json", "w"), indent=1)


if __name__ == "__main__":
    main()
