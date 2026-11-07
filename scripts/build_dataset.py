#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把老师给的 4 个文件统一成一份标准 jsonl，并输出审计报告。

四源关系（已核实：文本逐条 600/600 完全一致）：
  1) 豆瓣短评_35556001_前30页（带标）.xlsx  [主表] 原始抓取：页码/昵称/星级/时间/IP/有用数/内容
  2) 同上 Sheet1                             [冗余] ID/情绪/星级/内容  —— 与源3 重复
  3) 豆瓣短评标注.xlsx                        [标签A] ID/情绪标注/内容  —— 100% 由星级映射生成
  4) 豆瓣短评标注1.json                       [标签A] id/label/text      —— 与源3 完全等价
  5) 豆瓣短评情感标注.json                    [标签B] id/content/records —— 独立的语义标注

标签A(rule)     = 星级规则：1-2星→负面，3星→中性，4-5星→正面（586条严格符合，14条星级未知者人工补）
标签B(semantic) = 逐条阅读文本后的语义判断，与星级仅中度一致（kappa 0.53）
"""
import json, os, argparse, collections
import openpyxl

SRC = "/root/uploads"
F_MAIN = os.path.join(SRC, "1793847121579562869-豆瓣短评_35556001_前30页（带标）.xlsx")
F_XLABLE = os.path.join(SRC, "1793847121635617670-豆瓣短评标注.xlsx")
F_J1 = os.path.join(SRC, "1793847121433163609-豆瓣短评标注1.json")
F_J2 = os.path.join(SRC, "1793847121497136060-豆瓣短评情感标注.json")

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_all():
    wb = openpyxl.load_workbook(F_MAIN, read_only=True)
    main = list(wb["豆瓣短评_35556001_前30页"].iter_rows(values_only=True))[1:]
    wb.close()

    wb = openpyxl.load_workbook(F_XLABLE, read_only=True)
    xlab = list(wb["Sheet1"].iter_rows(values_only=True))[1:]
    wb.close()

    j1 = json.load(open(F_J1, encoding="utf-8"))
    j2 = json.load(open(F_J2, encoding="utf-8"))

    A = {str(r["id"]): r["label"] for r in j1}
    B = {str(v["id"]): (v["records"].get("情感") or [None])[0] for v in j2.values()}
    C = {str(int(r[0])): r[1] for r in xlab}

    # 交叉校验：三套标签必须完全等价
    diffAC = [i for i in A if A[i] != C.get(i)]
    assert not diffAC, f"源3 与 源4 标签不一致: {diffAC[:5]}"

    rows = []
    for idx, r in enumerate(main, start=1):
        page, nick, star, t, ip, votes, content = r[0], r[1], r[2], r[3], r[4], r[5], r[6]
        sid = str(idx)
        rows.append({
            "id": idx,
            "page": page,
            "nick": nick,
            "star": str(star) if star is not None else "未知",
            "time": str(t) if t else None,
            "ip": ip,
            "votes": int(votes) if isinstance(votes, (int, float)) else votes,
            "text": content,
            "label_rule": A.get(sid),          # 星级弱监督
            "label_semantic": B.get(sid),      # 语义标注
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = load_all()
    path = os.path.join(args.out, "reviews_600.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 同时导出两份「单标签」训练视图
    for key, name in [("label_rule", "label_rule.jsonl"), ("label_semantic", "label_semantic.jsonl")]:
        p = os.path.join(args.out, name)
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"id": r["id"], "text": r["text"], "label": r[key]},
                                   ensure_ascii=False) + "\n")

    # ---- 审计报告 ----
    RULE = {"1星": "负面", "2星": "负面", "3星": "中性", "4星": "正面", "5星": "正面"}
    lines = []
    lines.append("# 数据集审计报告\n")
    lines.append(f"- 样本数：{len(rows)}")
    lines.append(f"- 文本唯一性：{len(set(r['text'].strip() for r in rows))} 唯一 / {len(rows)} 总")
    lines.append(f"- 文本长度：min {min(len(r['text']) for r in rows)}，max {max(len(r['text']) for r in rows)}，"
                 f"均值 {sum(len(r['text']) for r in rows)/len(rows):.0f} 字")
    lines.append("\n## 标签分布\n")
    lines.append("| 标签 | 弱监督A(星级) | 语义B |")
    lines.append("|---|---|---|")
    ca = collections.Counter(r["label_rule"] for r in rows)
    cb = collections.Counter(r["label_semantic"] for r in rows)
    for lab in ["正面", "中性", "负面"]:
        lines.append(f"| {lab} | {ca[lab]} | {cb[lab]} |")
    lines.append(f"| **合计** | **{sum(ca.values())}** | **{sum(cb.values())}** |")

    lines.append("\n## 星级 × 标注B 交叉表\n")
    lines.append("| 星级 | 正面 | 中性 | 负面 | 合计 |")
    lines.append("|---|---|---|---|---|")
    cross = collections.Counter((r["star"], r["label_semantic"]) for r in rows)
    for s in ["1星", "2星", "3星", "4星", "5星", "未知"]:
        v = [cross.get((s, l), 0) for l in ["正面", "中性", "负面"]]
        lines.append(f"| {s} | {v[0]} | {v[1]} | {v[2]} | {sum(v)} |")

    viol = [r for r in rows if r["star"] in RULE and r["label_rule"] != RULE[r["star"]]]
    agree = sum(1 for r in rows if r["label_rule"] == r["label_semantic"])
    lines.append(f"\n## 一致性\n")
    lines.append(f"- 标注A 严格由星级映射推导：违规 {len(viol)} 条")
    lines.append(f"- A 与 B 逐条一致：{agree}/{len(rows)} = {agree/len(rows):.1%}")
    lines.append(f"- 星级 = 「未知」：{sum(1 for r in rows if r['star']=='未知')} 条（标签A 为人工补标）")

    rp = os.path.join(args.out, "audit.md")
    open(rp, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n已写出: {path}\n       {rp}")


if __name__ == "__main__":
    main()
