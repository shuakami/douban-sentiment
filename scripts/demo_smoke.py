#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke-test the demo API with representative Chinese reviews."""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

SAMPLES = [
    ("binary", "这电影节奏太拖沓了，演技也浮夸，浪费了一个好故事。"),
    ("binary", "神作！每一帧都能当壁纸，剧情烧脑但结局封神，二刷更爽。"),
    ("3class", "这电影节奏太拖沓了，演技也浮夸，浪费了一个好故事。"),
    ("3class", "还行吧，场面是大，但剧情一般，属于看了也就那样。"),
    ("3class", "神作！每一帧都能当壁纸，剧情烧脑但结局封神，二刷更爽。"),
    ("3class", "以广东早茶发誓热烈超好看不好看以后吃不到好吃的早茶"),
]


def post(path, payload):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    health = json.loads(urllib.request.urlopen(BASE + "/api/health", timeout=10).read().decode())
    print("health:", health)
    for task, text in SAMPLES:
        t0 = time.time()
        out = post("/api/predict", {"text": text, "task": task})
        wall = (time.time() - t0) * 1000
        probs = ", ".join(f"{n}={p:.3f}" for n, p in zip(out["class_names"], out["probabilities"]))
        chans = ", ".join(f"{c['name']}" for c in out["channels"])
        print(f"[{task}] {out['label']} conf={out['confidence']:.3f} "
              f"latency={out['latency_ms']}ms wall={wall:.0f}ms")
        print(f"    {probs}")
        print(f"    channels: {chans}")


if __name__ == "__main__":
    main()
