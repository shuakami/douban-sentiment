#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify web/short_lexicon.py against base 600 + aug 48.

Rule: normalize(text) must EXACTLY equal a lexicon key to fire.
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "web"))
import short_lexicon as SL  # noqa: E402


def main():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "reviews_600.jsonl"), encoding="utf-8")]
    base = [r for r in rows if r.get("src") != "aug"]
    aug = [r for r in rows if r.get("src") == "aug"]

    print("== base conflicts (normalized exact hit) ==")
    n_conf = 0
    for r in base:
        k = SL.norm_short(r["text"])
        lab = SL.LEX.get(k) if k and len(k) <= SL.MAX_NORM_LEN else None
        if lab:
            ok = "OK " if lab == r["label_semantic"] else "BAD"
            if lab != r["label_semantic"]:
                n_conf += 1
            print(f"  {ok} {k!r:20s} lex={lab} label={r['label_semantic']}  text={r['text']!r}")
    print(f"base conflicts: {n_conf}")

    print("\n== aug coverage ==")
    hit = miss = bad = 0
    for r in aug:
        lab = SL.lookup(r["text"], "3class")
        if lab is None:
            # binary-only fallback check not needed; report raw lex hit too
            k = SL.norm_short(r["text"])
            raw = SL.LEX.get(k) if k and len(k) <= SL.MAX_NORM_LEN else None
            miss += 1
            print(f"  --- miss {r['text']!r} label={r['label_semantic']} (raw={raw})")
        else:
            hit += 1
            if lab != r["label_semantic"]:
                bad += 1
                print(f"  BAD {r['text']!r} lex={lab} label={r['label_semantic']}")
    print(f"aug hit={hit} miss={miss} bad={bad}")


if __name__ == "__main__":
    main()
