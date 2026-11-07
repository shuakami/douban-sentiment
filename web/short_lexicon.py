#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ultra-short-text guard lexicon.

Fires ONLY when the whole normalized input exactly equals a key — i.e. the
user typed a bare sentiment slang / short phrase with zero review context,
where the transformer has no signal and tends to collapse to 中性.

Verified against all 600 real Douban reviews: zero label conflicts
(see scripts/verify_lexicon.py). Normal-length reviews never reach this
path because exact-match fails.
"""
import re

NEG = """操你妈 操你娘 草 艹 nm nmsl 尼玛 滚 退退退 退 拉胯 放屁 敷衍 避雷 垃圾 无聊 恶心 翻车 烂 差评 狗屁 难看 不好看 删前 别看 扯淡 胡说八道 看不下去 无语 真垃圾 烂片 烂俗 败笔 别拍 无聊透了 看了想睡 没劲 俗套 烂大街
傻逼 去死 吐槽 拉完""".split()

POS = """yyds 绝绝子 优秀 必看 强 天花板 神仙 上头 闭眼入 好看 不错 推荐 安利 爱了 真香 神作 封神 爆笑 感动 泪目 好评 满分 好评如潮 佳作 神片 力荐 强推 入坑 神剧 高能 绝美 绝了 太棒了 极致 完美 宝藏 还不错 棒 赞""".split()

NEU = """一般 还行 一般般 马马虎虎 凑合 无感 没啥感觉 说不上来 平平 中规中矩 还可以 尚可 过得去 不怎么样 还行吧 不好不坏 无所谓 平平无奇 普通 不知道 不功不过 正常发挥 意料之中 各花入各眼 可以""".split()

LEX = {}
for w in NEG:
    LEX[w] = "负面"
for w in POS:
    LEX[w] = "正面"
for w in NEU:
    LEX[w] = "中性"

MAX_NORM_LEN = 10


def norm_short(t):
    t = (t or "").strip().lower()
    return re.sub(r"[^\w一-鿿]+", "", t, flags=re.UNICODE)


def lookup(text, task):
    """Return '正面'/'中性'/'负面' or None. task in {'3class','binary'}.

    Binary has no 中性 class, so neutral hits fall through to the model.
    """
    k = norm_short(text)
    if not k or len(k) > MAX_NORM_LEN:
        return None
    lab = LEX.get(k)
    if lab is None:
        return None
    if task == "binary" and lab == "中性":
        return None
    return lab
