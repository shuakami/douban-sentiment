#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score synthetic candidates with the deployed ensemble and keep only
self-consistent rows (ensemble argmax == generation label).

Usage: python scripts/score_synth.py --in data/synth_candidates.jsonl
       --out data/synth_kept.jsonl [--min-conf 0.4]
"""
import os, sys, json, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "web"))
sys.path.insert(0, HERE)

import app as webapp  # noqa: E402

N3 = ["正面", "中性", "负面"]


def neu_p_neu_batch(ne, texts):
    p_stu = webapp.predict_channel_probs(ne["student"], texts)[:, 1]
    feats = webapp.backbone_feats(ne["head"], texts)
    device = next(ne["head"]["bb"]["model"].parameters()).device
    x = torch.tensor((feats - ne["head"]["mean"]) / ne["head"]["scale"],
                     dtype=torch.float32, device=device)
    with torch.inference_mode():
        logits = ne["head"]["net"](x).float()
    p_head = torch.softmax(logits, dim=-1).cpu().numpy()[:, 1]
    return (p_stu + p_head) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/synth_candidates.jsonl")
    ap.add_argument("--out", default="data/synth_kept.jsonl")
    ap.add_argument("--min-conf", type=float, default=0.0)
    args = ap.parse_args()

    webapp.startup()
    cands = [json.loads(l) for l in open(os.path.join(ROOT, args.inp), encoding="utf-8")]
    import re
    for c in cands:
        c["text"] = re.sub(r"^\s*\[[^\]]*\]\s*", "", c["text"]).strip()
    texts = [c["text"] for c in cands]
    print(f"scoring {len(texts)} candidates...", flush=True)

    BS = 32
    all_probs = []
    for i in range(0, len(texts), BS):
        chunk = texts[i:i + BS]
        ch = webapp.channel_probs("3class", chunk)
        base = webapp.weighted_mean(ch)
        ne = webapp.STATE["tasks"]["3class"]["neu_expert"]
        pn = neu_p_neu_batch(ne, chunk).reshape(-1, 1)
        posneg = base[:, [0, 2]]
        posneg = posneg / posneg.sum(1, keepdims=True)
        spec = np.concatenate([(1 - pn) * posneg[:, :1], pn, (1 - pn) * posneg[:, 1:]], axis=1)
        probs = (base + ne["weight"] * spec) / (1 + ne["weight"])
        all_probs.append(probs)
        print(f"  {i + len(chunk)}/{len(texts)}", flush=True)
    probs = np.concatenate(all_probs)

    kept, dropped = [], {"agree": 0, "conf": 0}
    per_class = {}
    for c, p in zip(cands, probs):
        gi = N3.index(c["gen_label"])
        pi = int(p.argmax())
        conf = float(p.max())
        per_class.setdefault(c["gen_label"], [0, 0])
        per_class[c["gen_label"]][1] += 1
        if pi == gi and conf >= args.min_conf:
            kept.append({"text": c["text"], "label_semantic": c["gen_label"],
                         "gen_conf": round(conf, 4), "gen_probs": [round(float(x), 4) for x in p]})
            per_class[c["gen_label"]][0] += 1
        elif pi == gi:
            dropped["conf"] += 1
        else:
            dropped["agree"] += 1

    for k, (a, b) in per_class.items():
        print(f"  {k}: kept {a}/{b}")
    print(f"total kept={len(kept)} dropped_disagree={dropped['agree']} dropped_conf={dropped['conf']}")

    with open(os.path.join(ROOT, args.out), "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
