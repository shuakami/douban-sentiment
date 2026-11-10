#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TEXT-LEVEL contamination audit for data/human400_labels.jsonl.

  exact   : stripped raw-text equality
  norm    : lowercase + drop Unicode cat P*/Z*/Cc/Cf/Cs
  neardup : Jaccard > 0.8 over char 3-gram sets of normalized text
            (index restricted to norm len>=12: shorter texts can only
            exceed 0.8 via identical norm, already covered by norm match)
Read-only. Runtime ~15-30 min (911K corpus scan + shingle index).
"""
import json, pickle, sys, unicodedata, hashlib
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

REPO = Path(r"D:\Project\douban-sentiment-package\douban-sentiment")
DATA = Path(r"E:\GB77_data")
H4   = REPO / "data" / "human400_labels.jsonl"
PKL  = DATA / "clean" / "comments.pkl"
JTHR, NG, DF_CAP = 0.8, 3, 50000
MIN_NORM_LEN = 12   # len<12 texts can only J>0.8-match identical norm
SELF = {H4, DATA/"human400"/"human400_labels.jsonl",
        DATA/"human400"/"human400_repo.jsonl",
        DATA/"human400"/"qwen_human400_labels.jsonl"}

def norm(s):
    return "".join(c for c in (s or "").lower()
                   if not (unicodedata.category(c)[0] in "PZ"
                           or unicodedata.category(c) in ("Cc","Cf","Cs")))

def shingles_norm(s):          # s must already be normalized
    n = len(s)
    if n == 0: return frozenset()
    if n <= NG: return frozenset([s])
    return frozenset(s[i:i+NG] for i in range(n-NG+1))

def hsh(g):
    return int.from_bytes(hashlib.blake2b(g.encode("utf-8"),
                        digest_size=8).digest(), "little")

h4 = [r for r in (json.loads(l) for l in open(H4, encoding="utf-8")) if r.get("label")]
assert len(h4) == 400
h4_raw  = {r["i"]: r["text"].strip() for r in h4}
h4_cid  = {r["i"]: str(r["cid"]) for r in h4}
h4_n2i  = defaultdict(list)
h4_nm   = {}
for r in h4:
    nm = norm(r["text"]); h4_nm[r["i"]] = nm
    h4_n2i[nm].append(r["i"])

print(f"[h4] {len(h4)} rows; norm-len<12 (norm-match only): "
      f"{sum(1 for i in h4_nm if len(h4_nm[i])<MIN_NORM_LEN)}", flush=True)

df = pd.read_pickle(PKL)
cid2content = dict(zip(df["cid"].astype(str),
                       df["content"].fillna("").astype(str)))

pools = {}   # name -> {"rows":int, "norm":{nm:[(cid,raw_strip)]}}
def add_pool(name, rows_iter):
    d, n = defaultdict(list), 0
    for cid, raw in rows_iter:
        if raw is None: continue
        raw = str(raw).strip(); n += 1
        d[norm(raw)].append((str(cid), raw))
    pools[name] = {"rows": n, "norm": d}
    print(f"[pool] {name}: rows={n} uniq_norm={len(d)}", flush=True)

def jtexts(path):
    out = []
    for l in open(path, encoding="utf-8"):
        l = l.strip()
        if not l: continue
        try: d = json.loads(l)
        except Exception: continue
        t = next((d[k] for k in ("text","content","review") if d.get(k)), None)
        c = next((d[k] for k in ("cid","id") if d.get(k) is not None), "?")
        if t is not None: out.append((c, t))
    return out

for f in sorted((REPO/"data").rglob("*.jsonl")):
    if f in SELF or f.name == "human400_labels.jsonl": continue
    add_pool("repo:"+str(f.relative_to(REPO)), jtexts(f))
for f in sorted((DATA/"lab2000").rglob("*.jsonl")):
    if f in SELF: continue
    add_pool("lab2000:"+str(f.relative_to(DATA/"lab2000")), jtexts(f))

def meta_rows(p):
    m = pickle.load(open(p, "rb"))
    if not hasattr(m, "columns") or "cid" not in m.columns: return None, []
    if "content" in m.columns: tx = m["content"]
    elif "text" in m.columns:     tx = m["text"]
    else: tx = m["cid"].astype(str).map(cid2content)
    return len(m), list(zip(m["cid"].astype(str), tx.fillna("").astype(str)))

for sub in ("pseudo", "features"):
    for p in sorted((DATA/sub).glob("*_meta.pkl")):
        nr, rows = meta_rows(p)
        if nr is None: continue
        if nr > 500000:
            print(f"[pool] {sub}:{p.name}: FULL-CORPUS manifest ({nr}) — skipped "
                  f"(self-corpus; covered by comments.pkl cross-row scan)", flush=True)
            continue
        add_pool(f"{sub}:{p.name}", rows)

def _sample(m, a, val_split):
    pc  = int(a.get("per_class", 0)); n = int(a.get("n", 200000))
    sd  = int(a.get("seed", 42))
    if pc: m = pd.concat([g.sample(min(len(g), pc), random_state=sd)
                          for _, g in m.groupby("y")])
    if n and len(m) > n: m = m.sample(n, random_state=sd)
    if val_split:                      # first 5000 of frac-shuffle = corpus-val
        m = m.sample(frac=1.0, random_state=sd)
        m = m.iloc[min(5000, len(m)//20):]
    return m

def rating_pool(a, val_split):
    d = df[df["len"] >= int(a.get("min_len", 4))]
    r = d["rating"].astype(int)
    rm = a.get("rmap") or ("tri" if a.get("task") == "3class" else "bin_strict")
    if rm == "bin_strict": m = d[(r<=2)|(r==5)].copy(); m["y"]=(m["rating"].astype(int)==5).astype(int)
    elif rm == "bin_all":   m = d[(r<=2)|(r>=4)].copy();   m["y"]=(m["rating"].astype(int)>=4).astype(int)
    else:                   m = d.copy();                 m["y"]=np.select([r>=4,r==3,r<=2],[0,1,2])
    return _sample(m, a, val_split)

def pseudo_pool(a, val_split):
    tag = a["pseudo"]; pdir = DATA/"pseudo"
    meta = pd.read_pickle(pdir/f"{tag}_meta.pkl")
    probs = np.load(pdir/f"{tag}_probs.npy")
    cf = probs.max(1); m = meta.copy(); m["y"] = probs.argmax(1)
    if a.get("task") == "neu": m["y"] = (m["y"] == 1).astype(int)
    if float(a.get("conf", 0)) > 0: m = m[cf >= float(a["conf"])].copy()
    if "content" not in m.columns:
        m["content"] = m["cid"].astype(str).map(cid2content)
    return _sample(m, a, val_split)

for kind, paths, vs in (("cfg",   sorted((DATA/"models").glob("*.json")),  True),
                        ("tfidf", sorted((DATA/"results").glob("*.json")), False)):
    for f in paths:
        if f.name.endswith("_goldmeta.json"): continue
        try: j = json.load(open(f, encoding="utf-8"))
        except Exception: continue
        a = j.get("args") or {}
        if not a or ("pseudo" not in a and "rmap" not in a and "task" not in a): continue
        try: m = pseudo_pool(a, vs) if a.get("pseudo") else rating_pool(a, vs)
        except Exception as e: print(f"[cfg] {f.name}: SKIP {e}", flush=True); continue
        exp = j.get("n_train") or j.get("n")
        ok  = "OK" if exp is None or int(exp) == len(m) else f"MISMATCH exp={exp}"
        print(f"[cfg] {kind}:{j.get('tag') or f.stem}: train_rows={len(m)} {ok}", flush=True)
        add_pool(f"{kind}:{j.get('tag') or f.stem}",
                 zip(m["cid"].astype(str), m["content"]))

# ---------- corpus cross-row exact/norm (self cid excluded) ----------
print("\n=== comments.pkl cross-row (non-self cid) ===", flush=True)
corp = defaultdict(list)
for cid, content in zip(df["cid"].astype(str), df["content"].fillna("").astype(str)):
    r = str(content).strip(); nm = norm(content)
    for i in h4_n2i.get(nm, ()):
        if cid != h4_cid[i]: corp[i].append((cid, r, r == h4_raw[i]))
ce = sum(1 for v in corp.values() for _,_,ex in v if ex)
cn = sum(1 for v in corp.values() for _,_,ex in v if not ex)
print(f"other-cid EXACT={ce}  other-cid NORM-ONLY={cn}  "
      f"h4_rows_affected={len(corp)}", flush=True)
for i, hits in list(corp.items())[:10]:
    print("  h4 i=%d cid=%s text=%.40s <- %s" % (i, h4_cid[i], h4_raw[i], hits[:2]), flush=True)

# ---------- exact + normalized per pool ----------
print("\n=== EXACT / NORMALIZED per pool (distinct h4 rows) ===", flush=True)
exact_hits, norm_hits = defaultdict(set), defaultdict(set)
for name, p in pools.items():
    for i in range(len(h4)):
        nm = h4_nm[i]
        for cid, raw in p["norm"].get(nm, ()):
            norm_hits[name].add(i)
            if raw == h4_raw[i]:
                exact_hits[name].add(i); break
    if exact_hits[name] or norm_hits[name]:
        print(f"{name:60s} exact={len(exact_hits[name])} norm={len(norm_hits[name])}", flush=True)
        for i in sorted(exact_hits[name])[:3]:
            print(f"    EXACT h4 i={i} cid={h4_cid[i]} text={h4_raw[i][:50]!r}", flush=True)

# ---------- near-duplicate: Jaccard>0.8 char-3gram ----------
print("\n=== building near-dup index (norm len>=%d only) ===" % MIN_NORM_LEN, flush=True)
uidx, texts, tpools = {}, [], []
for name, p in pools.items():
    for nm in p["norm"]:
        if len(nm) < MIN_NORM_LEN: continue
        j = uidx.get(nm)
        if j is None:
            uidx[nm] = len(texts); texts.append(nm); tpools.append(set())
        tpools[uidx[nm]].add(name)
print(f"unique pool texts indexed: {len(texts)}", flush=True)

post = defaultdict(list)
for j, t in enumerate(texts):
    for g in shingles_norm(t):
        hh = hsh(g); lst = post[hh]
        if len(lst) < DF_CAP: lst.append(j)
print("index built", flush=True)

sch_cache = {}
nd_hits = defaultdict(lambda: defaultdict(list))   # pool -> h4_i -> [(J, text)]
nd_examples = {}
for r in h4:
    i = r["i"]; nm = h4_nm[i]
    if len(nm) < MIN_NORM_LEN: continue
    S = shingles_norm(nm)
    cand = Counter()
    for g in S:
        for j in post.get(hsh(g), ()): cand[j] += 1
    need = JTHR * len(S)
    for j, shared in cand.items():
        if shared <= need: continue           # inter<=shared -> J<=0.8 impossible
        SC = sch_cache.get(j)
        if SC is None: SC = sch_cache[j] = shingles_norm(texts[j])
        inter = len(S & SC); uni = len(S | SC)
        J = inter / uni if uni else 0.0
        if J > JTHR:
            for pn in tpools[j]:
                nd_hits[pn][i].append((round(J, 3), texts[j]))
            nd_examples.setdefault(i, texts[j])

print("\n=== NEAR-DUP hits (J>0.8, distinct h4 rows) ===", flush=True)
for pn in sorted(nd_hits):
    rows_hit = nd_hits[pn]
    mx = max(v[0][0] for v in rows_hit.values())
    print(f"{pn:60s} h4rows={len(rows_hit)} maxJ={mx}", flush=True)
    for i in sorted(rows_hit)[:3]:
        print(f"    i={i} cid={h4_cid[i]} J={rows_hit[i][0][0]} "
              f"h4={h4_raw[i][:36]!r} pool={rows_hit[i][0][1][:36]!r}", flush=True)

# ---------- verdict ----------
tot_ex = set().union(*exact_hits.values()) if exact_hits else set()
tot_nm = set().union(*norm_hits.values()) if norm_hits else set()
tot_nd = set(nd_hits.keys()) and set().union(*[set(v) for v in nd_hits.values()]) or set()
print("\n=== VERDICT INPUTS ===")
print(f"exact h4 rows (any pool): {sorted(tot_ex)}")
print(f"norm-only extra rows:     {sorted(tot_nm - tot_ex)}")
print(f"neardup h4 rows (any pool): {sorted(tot_nd)}")
print(f"corpus other-cid hits:    {sorted(corp.keys())}")
