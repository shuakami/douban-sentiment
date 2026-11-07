#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成阿里云 NLP 自学习平台可直接上传的数据文件。

平台要求（官方文档 162018 / 609364）：
  - UTF-8 编码的 JSON
  - 结构：{"0": {"id":"1", "content":"文本", "records": {"情感": ["正面"]}}, ...}
  - records 的 value 可以是字符串（单标签）或数组（多标签）
  - 情感任务官方支持 正 / 中 / 负 三类，但建议每类至少 500 条

本脚本输出三份，对应三种打法：
  aliyun_3class.json   三分类原样（正面/中性/负面）—— 直接传，但样本量不足
  aliyun_binary.json   二分类（丢中性）—— 最容易跑出高分
  aliyun_merged.json   二分类（中性并入，按星级方向）—— 保住全部样本

用法：python3.11 to_aliyun_format.py
"""
import json, os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "reviews_600.jsonl")
OUTD = os.path.join(ROOT, "data", "aliyun")


def norm(lab):
    """统一成平台认的写法：正/中/负"""
    return {"正面": "正", "中性": "中", "负面": "负"}.get(lab, lab)


def dump(rows, path):
    obj = {}
    for i, r in enumerate(rows):
        obj[str(i)] = {
            "id": str(r["id"]),
            "content": r["text"],
            "records": {"情感": [r["_lab"]]},
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    c = collections.Counter(r["_lab"] for r in rows)
    print(f"  {os.path.basename(path):<22} {len(rows):>4} 条  {dict(c)}")


def main():
    os.makedirs(OUTD, exist_ok=True)
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8") if json.loads(l)["label_semantic"]]

    # 1) 三分类原样（用语义标注 B，比星级映射 A 干净）
    r3 = [dict(r, _lab=norm(r["label_semantic"])) for r in rows]
    print("生成阿里云自学习平台数据：")
    dump(r3, os.path.join(OUTD, "aliyun_3class.json"))

    # 2) 二分类，丢弃中性
    r2 = [dict(r, _lab=norm(r["label_semantic"])) for r in rows if r["label_semantic"] != "中性"]
    dump(r2, os.path.join(OUTD, "aliyun_binary.json"))

    # 3) 二分类，中性按星级方向并入（3星中自觉偏负的多，这里保守：归入正面侧前先看星级）
    r2b = []
    for r in rows:
        lab = r["label_semantic"]
        if lab == "中性":
            lab = "正面" if r["star"] in ("4星", "5星") else "负面"
        r2b.append(dict(r, _lab=norm(lab)))
    dump(r2b, os.path.join(OUTD, "aliyun_merged.json"))

    print("\n说明：平台官方建议每类 >=500 条，当前数据量在三分类下严重不足（负类仅 100 条）。")
    print("     若走平台路线，优先用 aliyun_binary.json（正 316 / 负 192）。")


if __name__ == "__main__":
    main()
