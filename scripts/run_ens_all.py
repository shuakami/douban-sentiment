#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run ensemble_eval over every matching OOF channel file in out/."""
import os, sys, glob, json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")
sys.path.insert(0, HERE)
import ensemble_eval as ee  # noqa: E402

PATS = {
    "binary": ["erlangshen*bin*.json", "macbert*bin*.json", "tfidf*bin*.json",
               "zs*bin*.json", "frozen*bin*.json", "swp*bin*.json", "lora*bin*.json"],
    "3class": ["erlangshen*3class*.json", "macbert*3class*.json",
               "tfidf*3class*.json", "frozen*3class*.json", "swp*3class*.json",
               "lora*3c*.json"],
}
EXCLUDE = ("ens_",)  # never feed ensemble outputs back as channels


def collect(task):
    seen = {}
    for pat in PATS[task]:
        for f in glob.glob(os.path.join(OUT, pat)):
            b = os.path.basename(f)
            if b.startswith(EXCLUDE):
                continue
            # keep first occurrence; verify task inside file
            try:
                r = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            if r.get("task") != task:
                continue
            if "probs" not in r or "y" not in r:
                continue
            seen[b] = f
    return [seen[k] for k in sorted(seen)]


def main():
    task = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else f"ens_{task}_allch"
    files = collect(task)
    print(f"# {task}: {len(files)} channels")
    for f in files:
        print("  ", os.path.basename(f))
    sys.argv = ["ensemble_eval.py"] + files + ["--task", task, "--tag", tag,
                                              "--greedy_iters", "60"]
    ee.main()


if __name__ == "__main__":
    main()
