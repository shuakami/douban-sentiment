#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FastAPI backend for the Douban sentiment demo.

Serves the compiled frontend and runs the deployed ensemble: distilled
MacBERT students + frozen MegatronBERT-1.3B MLP heads, plus a neutral
expert blend and temperature scaling for the 3-class task.
"""
import os, sys, json, time, argparse
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

WEB = Path(__file__).resolve().parent
ROOT = WEB.parent
SCRIPTS = ROOT / "scripts"
OUT = ROOT / "out"
sys.path.insert(0, str(SCRIPTS))
import gpu_finetune as gf  # noqa: E402
import frozen_megabert as fm  # noqa: E402
import distill_head as dh  # noqa: E402
import short_lexicon as sl  # noqa: E402
import joblib  # noqa: E402

app = FastAPI(title="Douban Sentiment Demo")
STATE = {"ready": False, "device": "cpu", "tasks": {}, "examples": {},
         "backbones": {}, "started": time.time()}
N3 = ["正面", "中性", "负面"]


def load_manifest():
    return json.loads((WEB / "manifest.json").read_text(encoding="utf-8"))


def build_tfidf(task):
    rows = gf.load_rows("label_semantic")
    texts, y = gf.make_labels(rows, task)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=2,
                          sublinear_tf=True, max_features=200000)
    clf = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced")
    pipe = Pipeline([("v", vec), ("c", clf)])
    pipe.fit(texts, y)
    return pipe


def load_frozen_backbone(model_id, device):
    """Load (or reuse) a frozen AutoModel backbone shared across tasks/heads."""
    if model_id not in STATE["backbones"]:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device).eval()
        STATE["backbones"][model_id] = {"tok": tok, "model": model}
    return STATE["backbones"][model_id]


def load_task(task, cfg, device):
    models = []
    for m in cfg.get("models", []):
        w = float(m.get("weight", 1.0))
        if m.get("type") == "frozen_lr":
            bb = load_frozen_backbone(m["model"], device)
            heads = []
            for h in m.get("heads", []):
                blob = joblib.load(ROOT / h["head"])
                heads.append((h.get("name") or Path(h["head"]).stem,
                              blob["pipeline"], blob.get("slice", "full"),
                              float(h.get("weight", w))))
            models.append({"kind": "frozen", "bb": bb, "heads": heads,
                           "max_len": int(m.get("max_len", 256))})
            continue
        if m.get("type") == "mlp_head":
            bb = load_frozen_backbone(m["model"], device)
            blob = torch.load(ROOT / m["head"], map_location="cpu", weights_only=False)
            net = dh.MLP(int(blob["in_dim"]), int(blob["n_classes"]), int(blob["hidden"]))
            net.load_state_dict(blob["state_dict"])
            net.to(device).eval()
            models.append({"kind": "mlp_head", "bb": bb, "net": net,
                           "mean": blob["scaler_mean"], "scale": blob["scaler_scale"],
                           "max_len": int(m.get("max_len", 256)), "weight": w,
                           "name": m.get("name") or Path(m["head"]).stem})
            continue
        path = ROOT / m["path"]
        tok = AutoTokenizer.from_pretrained(str(path))
        model = AutoModelForSequenceClassification.from_pretrained(str(path)).to(device).eval()
        models.append({"kind": "transformer", "path": str(path), "tok": tok, "model": model,
                       "max_len": int(m.get("max_len", 128)), "weight": w,
                       "name": m.get("name") or Path(m["path"]).name})
    neu = None
    if cfg.get("neu_expert"):
        ne = cfg["neu_expert"]
        sp = ROOT / ne["student"]["path"]
        stok = AutoTokenizer.from_pretrained(str(sp))
        smodel = AutoModelForSequenceClassification.from_pretrained(str(sp)).to(device).eval()
        student = {"kind": "transformer", "path": str(sp), "tok": stok, "model": smodel,
                   "max_len": int(ne["student"].get("max_len", 256))}
        hc = ne["head"]
        bb = load_frozen_backbone(hc["model"], device)
        blob = torch.load(ROOT / hc["head"], map_location="cpu", weights_only=False)
        hnet = dh.MLP(int(blob["in_dim"]), int(blob["n_classes"]), int(blob["hidden"]))
        hnet.load_state_dict(blob["state_dict"])
        hnet.to(device).eval()
        head = {"kind": "mlp_head", "bb": bb, "net": hnet,
                "mean": blob["scaler_mean"], "scale": blob["scaler_scale"],
                "max_len": int(hc.get("max_len", 256))}
        neu = {"student": student, "head": head, "weight": float(ne.get("weight", 0.5)),
               "student_w": float(ne.get("student_w", 0.5)),
               "head_w": float(ne.get("head_w", 0.5))}
    tfidf = build_tfidf(task) if cfg.get("tfidf") else None
    return {"models": models, "tfidf": tfidf, "neu_expert": neu,
            "temperature": float(cfg.get("temperature", 0.0)),
            "n_classes": len(gf.N3) if task == "3class" else 2}


def predict_channel_probs(entry, texts):
    device = next(entry["model"].parameters()).device
    amp = "bf16" if device.type == "cuda" else "none"
    return gf.predict_probs(entry["model"], entry["tok"], texts, entry["max_len"], device, 32, amp)


def backbone_feats(entry, texts):
    """Frozen-backbone features, cached so multiple heads share one pass."""
    device = next(entry["bb"]["model"].parameters()).device
    ck = (id(entry["bb"]["model"]), entry["max_len"], tuple(texts))
    cache = STATE.setdefault("feat_cache", {})
    if ck not in cache:
        if len(cache) > 8:
            cache.clear()
        cache[ck] = fm.extract_features(entry["bb"]["model"], entry["bb"]["tok"],
                                        texts, entry["max_len"], device, 16)
    return cache[ck]


def mlp_head_probs(entry, texts):
    device = next(entry["bb"]["model"].parameters()).device
    feats = backbone_feats(entry, texts)
    x = torch.tensor((feats - entry["mean"]) / entry["scale"],
                     dtype=torch.float32, device=device)
    with torch.inference_mode():
        logits = entry["net"](x).float()
    return torch.softmax(logits, dim=-1).cpu().numpy()


def channel_probs(task, texts):
    """Return [(name, probs ndarray, weight)] for every channel in the ensemble."""
    t = STATE["tasks"][task]
    out = []
    for entry in t["models"]:
        if entry.get("kind") == "frozen":
            feats = backbone_feats(entry, texts)
            half = feats.shape[1] // 2
            for name, pipe, sl, w in entry["heads"]:
                f = feats if sl == "full" else (feats[:, :half] if sl == "cls" else feats[:, half:])
                out.append((name, pipe.predict_proba(f), w))
        elif entry.get("kind") == "mlp_head":
            out.append((entry["name"], mlp_head_probs(entry, texts), entry["weight"]))
        else:
            out.append((entry["name"], predict_channel_probs(entry, texts), entry.get("weight", 1.0)))
    if t["tfidf"] is not None:
        out.append(("tfidf", t["tfidf"].predict_proba(texts), 1.0))
    return out


def weighted_mean(ch):
    ws = np.array([w for _, _, w in ch], dtype=np.float64)
    return np.average(np.stack([p for _, p, _ in ch]), axis=0, weights=ws)


def ensemble_probs(task, texts):
    return weighted_mean(channel_probs(task, texts))


def neu_p_neu(ne, text):
    """P(neutral) from the two neutral specialists, shape (1,)."""
    texts = [text]
    p_stu = predict_channel_probs(ne["student"], texts)[:, 1]
    feats = backbone_feats(ne["head"], texts)
    device = next(ne["head"]["bb"]["model"].parameters()).device
    x = torch.tensor((feats - ne["head"]["mean"]) / ne["head"]["scale"],
                     dtype=torch.float32, device=device)
    with torch.inference_mode():
        logits = ne["head"]["net"](x).float()
    p_head = torch.softmax(logits, dim=-1).cpu().numpy()[:, 1]
    sw, hw = ne.get("student_w", 0.5), ne.get("head_w", 0.5)
    return (sw * p_stu + hw * p_head) / (sw + hw)


def to_display(task, probs, threshold=0.5):
    """Return user-facing class names, aligned probs, pred label and confidence."""
    if task == "3class":
        names = N3
        disp = probs
        pred_idx = int(np.argmax(probs, axis=1)[0]) if probs.ndim == 2 else int(np.argmax(probs))
    else:
        # model order is [负面, 正面]; show [正面, 负面]
        names = ["正面", "负面"]
        disp = np.stack([probs[:, 1], probs[:, 0]], axis=1) if probs.ndim == 2 else np.array([probs[1], probs[0]])
        pos = disp[:, 0] if disp.ndim == 2 else np.array([disp[0]])
        pred_idx = int((pos < threshold).astype(int)[0])
    row = disp[0] if disp.ndim == 2 else disp
    return names, row.tolist(), names[pred_idx], float(row[pred_idx])


def load_examples():
    rows = gf.load_rows("label_semantic")
    ex = {"3class": [], "binary": []}
    # deterministic, diverse picks
    picks = {"正面": [], "中性": [], "负面": []}
    for r in rows:
        lab = r["label_semantic"]
        if len(picks[lab]) < 3:
            picks[lab].append(r)
    for lab in ["正面", "中性", "负面"]:
        for r in picks[lab]:
            ex["3class"].append({"id": r["id"], "label": lab, "star": r["star"], "text": r["text"]})
            if lab != "中性":
                ex["binary"].append({"id": r["id"], "label": lab, "star": r["star"], "text": r["text"]})
    STATE["examples"] = ex


@app.on_event("startup")
def startup():
    manifest = load_manifest()
    device = torch.device("cuda" if torch.cuda.is_available() and manifest.get("use_gpu", True) else "cpu")
    STATE["device"] = str(device)
    STATE["binary_threshold"] = float(manifest.get("binary_threshold", 0.5))
    print(f"[demo] device={device} loading manifest tasks={list(manifest['tasks'])}", flush=True)
    for task, cfg in manifest["tasks"].items():
        STATE["tasks"][task] = load_task(task, cfg, device)
        print(f"[demo] loaded {task}: {len(STATE['tasks'][task]['models'])} models tfidf={STATE['tasks'][task]['tfidf'] is not None}", flush=True)
    load_examples()
    STATE["ready"] = True
    print(f"[demo] ready in {time.time()-STATE['started']:.1f}s", flush=True)


class PredictReq(BaseModel):
    text: str
    task: str = "3class"


@app.get("/")
def index():
    return FileResponse(WEB / "static" / "index.html")


@app.get("/favicon.svg")
def favicon():
    return FileResponse(WEB / "static" / "favicon.svg", media_type="image/svg+xml")


@app.get("/icons.svg")
def icons():
    return FileResponse(WEB / "static" / "icons.svg", media_type="image/svg+xml")


@app.get("/assets/{file_path:path}")
def asset(file_path: str):
    base = (WEB / "static" / "assets").resolve()
    p = (base / file_path).resolve()
    if not str(p).startswith(str(base)) or not p.is_file():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/paper.pdf")
def paper():
    p = ROOT / "thesis" / "thesis.pdf"
    if not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, media_type="application/pdf",
                        filename="thesis.pdf",
                        content_disposition_type="inline")


@app.get("/paper_zh.pdf")
def paper_zh():
    p = ROOT / "thesis" / "thesis_zh.pdf"
    if not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, media_type="application/pdf",
                        filename="thesis_zh.pdf",
                        content_disposition_type="inline")


@app.get("/api/health")
def health():
    return {"ready": STATE["ready"], "device": STATE["device"],
            "tasks": {k: {"models": len(v["models"]), "tfidf": v["tfidf"] is not None}
                      for k, v in STATE["tasks"].items()},
            "uptime_s": time.time() - STATE["started"]}


@app.get("/api/examples")
def examples(task: str = "3class"):
    return {"task": task, "examples": STATE["examples"].get(task, [])[:9]}


@app.get("/api/metrics")
def metrics():
    def read_best(name):
        p = OUT / name
        if not p.exists():
            return None
        r = json.loads(p.read_text(encoding="utf-8"))
        best = r.get("best") or r.get("oof") or {}
        return {"acc": best.get("acc"), "macro_f1": best.get("macro_f1")}
    return {
        "binary": {"baseline": {"acc": 0.8406, "macro_f1": 0.8229},
                   "ours": read_best("ens_lite_binary.json") or read_best("ens_binary_allch2.json") or read_best("ens_binary_v6.json") or read_best("ens_binary_v4.json") or read_best("ens_binary_v2.json") or read_best("ens_binary_all.json")},
        "3class": {"baseline": {"acc": 0.7233, "macro_f1": 0.6175},
                   "ours": read_best("ens_lite_3class_v3.json") or read_best("ens_lite_3class_v2.json") or read_best("ens_lite_3class.json") or read_best("ens_3class_allch.json") or read_best("ens_3class_v5.json") or read_best("ens_3class_v4.json") or read_best("ens_3class_v3.json") or read_best("ens_3class_v2.json") or read_best("ens_3class_all.json")},
    }


@app.post("/api/predict")
def predict(req: PredictReq):
    if not STATE["ready"]:
        raise HTTPException(503, "model still loading")
    task = req.task if req.task in STATE["tasks"] else "3class"
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(400, "text is empty")
    if len(text) > 4000:
        text = text[:4000]
    t0 = time.time()
    lex = sl.lookup(text, task)
    if lex is not None:
        # ultra-short slang with no review context: high-precision rule hit,
        # skip the GPU ensemble entirely (verified 0 conflicts on 600 reviews)
        names = N3 if task == "3class" else ["正面", "负面"]
        idx = names.index(lex)
        disp = [0.03] * len(names)
        disp[idx] = 0.94
        return {
            "task": task,
            "text": text,
            "label": lex,
            "confidence": 0.94,
            "class_names": names,
            "probabilities": disp,
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "device": STATE["device"],
            "channels": [{"name": "超短文本规则", "weight": 1,
                          "probs": [round(x, 4) for x in disp]}],
        }
    ch = channel_probs(task, [text])
    probs = weighted_mean(ch)
    t = STATE["tasks"][task]
    spec = None
    if task == "3class" and t.get("neu_expert"):
        ne = t["neu_expert"]
        pn = neu_p_neu(ne, text).reshape(1, 1)
        posneg = probs[:, [0, 2]]
        posneg = posneg / posneg.sum(1, keepdims=True)
        spec = np.concatenate([(1 - pn) * posneg[:, :1], pn, (1 - pn) * posneg[:, 1:]], axis=1)
        w = ne["weight"]
        probs = (probs + w * spec) / (1 + w)
    temp = float(t.get("temperature", 0.0) or 0.0)
    if temp > 0 and probs.shape[1] > 1:
        logits = np.log(np.clip(probs, 1e-9, 1.0)) / temp
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        probs = e / e.sum(axis=1, keepdims=True)
    names, disp, label, conf = to_display(task, probs, STATE.get("binary_threshold", 0.5))
    dt = (time.time() - t0) * 1000
    channels = []
    for name, p, w in ch:
        row = p[0]
        if task == "binary":
            row = row[::-1]  # model order [负面, 正面] -> display [正面, 负面]
        channels.append({"name": name, "weight": w, "probs": np.round(row, 4).tolist()})
    if spec is not None:
        channels.append({"name": "中性专家", "weight": ne["weight"],
                         "probs": np.round(spec[0], 4).tolist()})
    return {
        "task": task,
        "text": text,
        "label": label,
        "confidence": round(conf, 4),
        "class_names": names,
        "probabilities": [round(x, 4) for x in disp],
        "latency_ms": round(dt, 1),
        "device": STATE["device"],
        "channels": channels,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
