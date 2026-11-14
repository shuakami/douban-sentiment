#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S4 — label the 5,000 random corpus reviews with DeepSeek-Flash.

Reads E:/GB77_data/lab2000/rand5k_pool.jsonl {"i","cid","text","rating"},
calls the SAME call_api() prompt/config as S1 (review text only, no rating
sent), writes E:/GB77_data/lab2000/extra_rand5k_ds.jsonl in the training
format {"text","rating","final"} plus keeps cid/i/raw for audit.

Resumable: rows already present (non-ERROR) are skipped; ERROR rows retried.
Unparseable rows retried once, then kept as final=null (counted as errors).
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, r"E:\GB77_data\lab2000")
from deepseek_label import call_api  # noqa: E402  (prompt lives here)

DATA_ROOT = Path(os.environ.get("DOUBAN_DATA_ROOT", r"E:\GB77_data"))
LAB = DATA_ROOT / "lab2000"
POOL = LAB / "rand5k_pool.jsonl"
OUT = LAB / "extra_rand5k_ds.jsonl"
RAW_OUT = LAB / "rand5k_ds_raw.jsonl"
MODEL = "deepseek-flash"


def load_done():
    done = set()
    if RAW_OUT.exists():
        for l in open(RAW_OUT, encoding="utf-8"):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if not str(r.get("raw", "")).startswith("ERROR"):
                done.add(int(r["i"]))
    return done


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    items = [json.loads(l) for l in open(POOL, encoding="utf-8")]
    done = load_done()
    todo = [it for it in items if int(it["i"]) not in done]
    print(f"[rand5k] done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return

    fout = open(RAW_OUT, "a", encoding="utf-8")
    second = []
    t0 = time.time()
    n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(call_api, it["text"], MODEL, "3class",
                          4096, "low"): it for it in todo}
        for fut in as_completed(futs):
            it = futs[fut]
            res = fut.result()
            rec = {"i": int(it["i"]), "cid": it["cid"], "label": res.get("label"),
                   "raw": res.get("raw"), "latency_s": res.get("latency_s")}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n += 1
            if res.get("label") is None and not str(res.get("raw", "")).startswith("ERROR"):
                second.append(it)
            if n % 200 == 0:
                print(f"  {n}/{len(todo)} elapsed={time.time()-t0:.0f}s", flush=True)
    if second:
        print(f"  retrying {len(second)} unparseable once", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(call_api, it["text"], MODEL, "3class",
                              4096, "low"): it for it in second}
            for fut in as_completed(futs):
                it = futs[fut]
                res = fut.result()
                fout.write(json.dumps({"i": int(it["i"]), "cid": it["cid"],
                                       "label": res.get("label"), "raw": res.get("raw"),
                                       "latency_s": res.get("latency_s")},
                              ensure_ascii=False) + "\n")
                fout.flush()
    fout.close()

    # build training file: join pool (text,rating) with latest non-ERROR label per i
    labels = {}
    for l in open(RAW_OUT, encoding="utf-8"):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if str(r.get("raw", "")).startswith("ERROR"):
            continue
        labels[int(r["i"])] = r.get("label")   # last write wins (retry overwrites)
    n_out = n_null = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for it in items:
            lab = labels.get(int(it["i"]))
            f.write(json.dumps({"text": it["text"],
                                "rating": it.get("rating"),
                                "final": lab if lab in ("正面", "中性", "负面") else None},
                               ensure_ascii=False) + "\n")
            n_out += 1
            n_null += lab not in ("正面", "中性", "负面")
    print(f"[rand5k] wrote {OUT} rows={n_out} null_labels={n_null} "
          f"elapsed={(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
