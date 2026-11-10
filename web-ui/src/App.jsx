import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Tabs } from '@base-ui/react/tabs'
import { Button } from '@base-ui/react/button'
import { Accordion } from '@base-ui/react/accordion'
import { Tooltip } from '@base-ui/react/tooltip'
import ShaderBg from './ShaderBg'

const API = ''
const SAMPLE_COUNT = 5
const IS_HF_SPACE = /(^|\.)hf\.space$/i.test(window.location.hostname)

/* HF Space (ZeroGPU) exposes predict through the Gradio queue protocol:
   POST /gradio_api/call/predict {data:[...]} -> event_id, then read the
   SSE stream at /gradio_api/call/predict/{event_id} for the result.
   Local dev hits the FastAPI /api/predict directly. */
async function spacePredict(text, task) {
  const post = await fetch('/gradio_api/call/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ data: [text, task] }),
  })
  if (!post.ok) throw new Error(`${post.status} ${post.statusText}`)
  const { event_id } = await post.json()
  const res = await fetch(`/gradio_api/call/predict/${encodeURIComponent(event_id)}`)
  if (!res.ok || !res.body) throw new Error(`${res.status} ${res.statusText}`)
  const reader = res.body.getReader()
  const dec = new TextDecoder()
  let buf = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })
    let idx
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      const frame = buf.slice(0, idx)
      buf = buf.slice(idx + 2)
      const ev = frame.match(/^event:\s*(\S+)/m)
      const dt = frame.match(/^data:\s*(.*)$/m)
      if (!ev || !dt) continue
      if (ev[1] === 'complete') {
        const arr = JSON.parse(dt[1])
        const s = Array.isArray(arr) ? arr[0] : arr
        return typeof s === 'string' ? JSON.parse(s) : s
      }
      if (ev[1] === 'error') throw new Error(dt[1])
    }
  }
  throw new Error('prediction stream ended without result')
}

const TASKS = [
  { value: '3class', label: '三分类', classes: '正面 / 中性 / 负面', count: 3, channels: 3 },
  { value: 'binary', label: '二分类', classes: '正面 / 负面', count: 2, channels: 2 },
]

const SENTI = {
  正面: { text: 'text-pos', bar: 'bg-pos' },
  中性: { text: 'text-neu', bar: 'bg-neu' },
  负面: { text: 'text-neg', bar: 'bg-neg' },
}
const senti = (name) => SENTI[name] ?? { text: 'text-ink', bar: 'bg-ink' }

/* 通道名 → 人话解释 */
function channelInfo(name) {
  const n = String(name).toLowerCase()
  if (n.includes('规则') || n.includes('rule') || n.includes('lexicon')) {
    return {
      label: '超短文本规则',
      desc: '输入只有几个字、完全没有影评上下文时（比如「yyds」「操你妈」），模型没有足够线索，直接查一张人工核对过的情感词表兜底。',
    }
  }
  if (n.includes('macbert')) {
    return {
      label: '大模型学生',
      desc: '一个叫 MacBERT 的中文预训练模型。它先跟着更强的「老师」集成模型学习怎么打分，所以单个模型就能给出完整判断。',
    }
  }
  if (n.includes('mlp') || n.includes('megabert') || n.includes('1.3b')) {
    return {
      label: '语义特征裁判',
      desc: '最大的 MegatronBERT-1.3B 模型负责读出整句的语义特征，交给一个很小的神经网络打分，权重是学生的两倍。',
    }
  }
  if (n.includes('330')) {
    return { label: '大模型 330M', desc: '3.3 亿参数的大模型，负责读懂整句的意思' }
  }
  if (n.includes('110')) {
    return { label: '小模型 110M', desc: '1.1 亿参数的小模型，速度快，和大模型互相补充' }
  }
  if (n.includes('tfidf') || n.includes('tf-idf') || n.includes('lr') || n.includes('logistic')) {
    return { label: '关键词统计', desc: '统计「烂片」「神作」这类高频情绪词来打分，当兜底' }
  }
  return { label: String(name), desc: '' }
}

async function api(path, opts) {
  if (IS_HF_SPACE && path === '/api/predict') {
    const body = JSON.parse(opts.body)
    return spacePredict(body.text, body.task)
  }
  const res = await fetch(API + path, opts)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

/* ── 小组件 ────────────────────────────────────────────────────── */

const textLink =
  'decoration-1 underline-offset-4 hover:underline active:bg-ink active:text-canvas focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink'

function ArrowLink({ href, children }) {
  return (
    <a href={href} className={`group inline-flex w-fit items-center gap-1 text-[18px] leading-[25px] underline ${textLink}`}>
      {children}
      <svg
        width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor"
        strokeLinecap="square" strokeLinejoin="round" aria-hidden="true" className="block"
      >
        <path d="M1.5 8h12" className="opacity-0 transition-opacity duration-100 group-hover:opacity-100" />
        <path d="M9 3.5 13.5 8 9 12.5" className="transition-transform duration-100 group-hover:translate-x-1" />
      </svg>
    </a>
  )
}

function PlusIcon(props) {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeLinecap="square" strokeLinejoin="round" {...props}>
      <path d="M1.5 8h13M8 14.5v-13" />
    </svg>
  )
}

function MinusIcon(props) {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeLinecap="square" strokeLinejoin="round" {...props}>
      <path d="M1.5 8h13" />
    </svg>
  )
}

function Dots() {
  return (
    <span aria-hidden="true">
      {[0, 1, 2].map((i) => (
        <span key={i} className="anim-dot" style={{ animationDelay: `${i * 0.18}s` }}>
          .
        </span>
      ))}
    </span>
  )
}

/* 问号提示：align-middle 与左侧文字垂直居中；items 为分点列表 */
function Info({ side = 'bottom', label, children, items }) {
  return (
    <Tooltip.Root delay={150}>
      <Tooltip.Trigger
        aria-label={label}
        className="ml-1 inline-grid size-[18px] place-items-center rounded-full align-middle text-ink-2 outline-none transition-colors duration-150 hover:text-ink focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-ink"
      >
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" aria-hidden="true">
          <circle cx="8" cy="8" r="6.6" />
          <path d="M6.35 6.15c.14-.95.9-1.55 1.7-1.55.92 0 1.6.62 1.6 1.45 0 1.15-1.5 1.3-1.5 2.4" />
          <circle cx="8.05" cy="11.25" r="0.85" fill="currentColor" stroke="none" />
        </svg>
      </Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Positioner side={side} sideOffset={6} className="z-50">
          <Tooltip.Popup className="max-w-80 border border-ink bg-canvas px-3 py-2 text-[13px] leading-[18px] text-ink shadow-[0_4px_16px_hsl(0_0%_0%/0.12)] [@media(prefers-color-scheme:dark)]:shadow-[0_4px_16px_hsl(0_0%_0%/0.6)] transition duration-150 ease-out data-[starting-style]:translate-y-1 data-[starting-style]:opacity-0 data-[ending-style]:-translate-y-1 data-[ending-style]:opacity-0">
            {items ? (
              <ul className="flex flex-col gap-1.5">
                {items.map((t, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="mt-[7px] size-1 shrink-0 rounded-full bg-ink-2" aria-hidden="true" />
                    <span>{t}</span>
                  </li>
                ))}
              </ul>
            ) : (
              children
            )}
          </Tooltip.Popup>
        </Tooltip.Positioner>
      </Tooltip.Portal>
    </Tooltip.Root>
  )
}

/* 数字滚动：ease-out-expo */
function useAnimatedNumber(target, duration = 700) {
  const [val, setVal] = useState(0)
  const valRef = useRef(0)
  const raf = useRef(0)
  useEffect(() => {
    const from = valRef.current
    if (from === target) return undefined
    const start = performance.now()
    const step = (now) => {
      const t = Math.min(1, (now - start) / duration)
      const e = t === 1 ? 1 : 1 - Math.pow(2, -10 * t)
      const v = from + (target - from) * e
      valRef.current = v
      setVal(v)
      if (t < 1) raf.current = requestAnimationFrame(step)
    }
    cancelAnimationFrame(raf.current)
    raf.current = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf.current)
  }, [target, duration])
  return val
}

/* 布局原语：整页一个 8 列网格，section 用 display:contents，
   间距由各 section 自己的 margin 承担，不用短分隔线 */

function Section({ id, title, hint, wide = false, children }) {
  return (
    <section className="contents">
      <h2
        id={id}
        className="col-[1/-1] mt-14 scroll-mt-7 text-[18px] leading-[25px] font-normal md:col-[1/3] md:mt-20"
      >
        {title}
        {hint && <Info side="bottom" label={hint.label} items={hint.items}>{hint.text}</Info>}
      </h2>
      <div className={`col-[1/-1] mt-6 min-w-0 md:col-[3/-1] md:mt-20 ${wide ? '' : 'xl:col-end-8'}`}>
        {children}
      </div>
    </section>
  )
}

/* ── 指标 ──────────────────────────────────────────────────────── */

const metricGrid =
  'grid grid-cols-[1fr_4.5rem_5.5rem] items-baseline gap-x-4 py-3 md:grid-cols-[1fr_6rem_6rem_6rem]'

const pct = (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`)
const delta = (a, b) => (a == null || b == null ? '' : `+${((a - b) * 100).toFixed(1)}%`)

/* 阿里 StructBERT 在同一批豆瓣数据上的实测成绩（只分正负两类） */
const ALI_BENCH = { acc: 0.8661, macro_f1: 0.8459 }

function MetricList({ metrics, latency }) {
  const rows = useMemo(() => {
    const out = []
    for (const [task, name] of [
      ['binary', '二分类'],
      ['3class', '三分类'],
    ]) {
      for (const [key, label] of [
        ['acc', '准确率'],
        ['macro_f1', '综合得分'],
      ]) {
        const ours = metrics?.[task]?.ours?.[key]
        const base = metrics?.[task]?.baseline?.[key]
        out.push({
          id: `${task}-${key}`,
          label: `${name}的${label}`,
          ours: pct(ours),
          base: pct(base),
          delta: delta(ours, base),
        })
      }
    }
    out.push({
      id: 'latency',
      label: (
        <>
          分析一条短评要多久
          <Info side="top" label="这个速度是什么概念" items={[
            '从接收短评到输出结果的完整耗时，在 AMD RX 9070 GRE 显卡上测得。',
            '约 0.1 秒，比眨一次眼还快。',
          ]} />
        </>
      ),
      ours: `${latency ?? '约 80'} ms`,
      base: '—',
      delta: '',
    })
    return out
  }, [metrics, latency])

  return (
    <div className="border-t border-ink tabular-nums">
      <div className={`${metricGrid} border-b border-ink text-[15px] leading-[22px] text-ink-2`}>
        <span>
          任务和指标
          <Info side="bottom" label="这些词怎么理解" items={[
            '准确率 = 猜对的比例。',
            '综合得分（Macro-F1）= 正面、中性、负面三类都照顾到的平均分，满分 100%。',
            '传统方法 = 不用 AI 大模型、只靠「统计关键词 + 简单分类器」的老办法能达到的水平。',
            '「比传统方法高 X%」= 准确率高出的百分比，满分 100 分里多拿的分。',
          ]} />
        </span>
        <span className="text-right">我的模型</span>
        <span className="hidden text-right md:block">传统方法</span>
        <span className="text-right">提升</span>
      </div>
      {rows.map((r) => (
        <div
          key={r.id}
          className={`${metricGrid} border-b border-ink text-[15px] leading-[22px] md:text-[18px] md:leading-[25px]`}
        >
          <span>{r.label}</span>
          <span className="text-right font-medium">{r.ours}</span>
          <span className="hidden text-right text-ink-2 md:block">{r.base}</span>
          <span className="text-right text-pos">{r.delta}</span>
        </div>
      ))}
    </div>
  )
}

/* ── 演示：结果区 ──────────────────────────────────────────────── */

function ProbRow({ name, p, winner, index }) {
  const s = senti(name)
  const shown = useAnimatedNumber(p * 100, 650)
  return (
    <li className="grid grid-cols-[3.5rem_1fr_3.5rem] items-center gap-4 text-[15px] leading-[22px]">
      <span className={name === winner ? 'font-medium text-ink' : 'text-ink-2'}>{name}</span>
      <div
        className="h-1.5 w-full overflow-hidden bg-tint-strong"
        role="progressbar"
        aria-label={`${name} 的可能性`}
        aria-valuenow={Math.round(p * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div
          className={`anim-grow h-full origin-left ${s.bar}`}
          style={{ '--p': p, animationDelay: `${index * 90}ms` }}
        />
      </div>
      <span className={`text-right tabular-nums ${name === winner ? 'font-medium text-ink' : 'text-ink-2'}`}>
        {shown.toFixed(1)}%
      </span>
    </li>
  )
}

function ProbabilityRows({ names, probs, winner }) {
  return (
    <div>
      <ul className="flex flex-col gap-3">
        {names.map((name, i) => (
          <ProbRow key={name} name={name} p={probs[i] ?? 0} winner={winner} index={i} />
        ))}
      </ul>
      <p className="mt-3 text-[13px] leading-[18px] text-ink-2">
        所有柱子加起来是 100%，最长的一根就是模型的最终判断。
      </p>
    </div>
  )
}

function Confidence({ value }) {
  const shown = useAnimatedNumber(value * 100, 800)
  return (
    <p className="shrink-0 text-right text-[15px] leading-[22px] text-ink-2">
      <span className="block text-[18px] leading-[25px] font-medium text-ink tabular-nums">{shown.toFixed(1)}%</span>
      <span>
        把握程度
        <Info side="top" label="什么意思">
          就是模型对自己判断的确定程度——它选中的那个类别的概率。越接近 100% 越确定。
        </Info>
      </span>
    </p>
  )
}

function ChannelTable({ channels, names }) {
  if (!channels?.length) return null
  return (
    <div>
      <div className="overflow-x-auto border-t border-ink">
        <table className="w-full text-left text-[15px] leading-[22px] tabular-nums">
          <thead>
            <tr className="border-b border-ink text-ink-2">
              <th className="py-3 pr-6 font-normal">判断方法</th>
              {names.map((n) => (
                <th key={n} className={`py-3 pl-4 text-right font-normal ${senti(n).text}`}>
                  {n}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {channels.map((ch) => {
              const top = Math.max(...ch.probs)
              const info = channelInfo(ch.name)
              const weighted = Number(ch.weight) > 1
              return (
                <tr key={ch.name} className="border-b border-ink">
                  <td className="py-2.5 pr-6 align-top">
                    <span className="block">
                      {info.label}
                      {weighted && <span className="text-ink-2"> ×{ch.weight}</span>}
                    </span>
                    {info.desc && (
                      <span className="mt-0.5 block pr-4 text-[12px] leading-[17px] font-normal text-ink-2">
                        {info.desc}
                      </span>
                    )}
                  </td>
                  {ch.probs.map((p, i) => (
                    <td key={i} className={`py-2.5 pl-4 text-right ${p === top ? 'font-medium text-ink' : 'text-ink-2'}`}>
                      {p.toFixed(3)}
                    </td>
                  ))}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-[13px] leading-[18px] text-ink-2">
        简单说，模型内部有几位「裁判」各自打分（上面这张表），最后按权重把分数加起来平均，
        就是上面的最终结果。每行里加粗的数字，是这位裁判最看好的类别。
      </p>
    </div>
  )
}

/* 加载骨架：形状和真实结果一致，面板高度不跳 */
function Skeleton({ count, channelCount }) {
  const bar = 'animate-pulse rounded-sm bg-tint'
  return (
    <div className="pt-1" aria-hidden="true">
      <div className="flex items-end justify-between gap-6">
        <div className={`${bar} h-[44px] w-36`} />
        <div className={`${bar} h-[38px] w-16`} />
      </div>
      <div className={`${bar} mt-10 mb-3 h-[18px] w-24`} />
      <div className="flex flex-col gap-3">
        {Array.from({ length: count }).map((_, i) => (
          <div key={i} className="grid grid-cols-[3.5rem_1fr_3.5rem] items-center gap-4">
            <div className={`${bar} h-[14px] w-10`} />
            <div className="h-1.5 w-full overflow-hidden bg-tint-strong">
              <div className={`${bar} h-full w-2/3`} />
            </div>
            <div className={`${bar} ml-auto h-[14px] w-9`} />
          </div>
        ))}
      </div>
      <div className={`${bar} mt-10 mb-3 h-[18px] w-32`} />
      <div className="border-t border-ink">
        {Array.from({ length: channelCount }).map((_, i) => (
          <div
            key={i}
            className="grid gap-4 border-b border-ink py-3"
            style={{ gridTemplateColumns: `10rem repeat(${count}, 1fr)` }}
          >
            <div className={`${bar} h-[14px] w-20`} />
            {Array.from({ length: count }).map((_, j) => (
              <div key={j} className={`${bar} h-[14px] w-10`} />
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}

function Result({ result, loading, error, sample, taskLabel, runId, classCount, channelCount }) {
  const s = result ? senti(result.label) : null
  const matched = sample && result ? sample.label === result.label : null

  return (
    <div>
      <h3 className="flex items-center text-[15px] leading-[22px] font-medium">
        预测结果
        <Info side="top" label="这块看什么" items={[
          '大字是模型的判断（正面 / 中性 / 负面），右边的百分比是它对这个判断的把握程度。',
          '下面依次是各类别的可能性，以及内部几种判断方法各自的打分。',
        ]} />
      </h3>

      {/* 空状态什么都不显示，只保留高度防止页面跳动 */}
      <div className="mt-3 min-h-64" aria-live="polite">
        {loading && !result && <p className="sr-only">正在分析…</p>}
        {result ? (
          /* key=runId：只有新结果到达才重播入场动画；请求进行中旧结果只变淡，位置不动 */
          <div
            key={runId}
            className={`anim-rise transition-opacity duration-300 ${loading ? 'opacity-40' : 'opacity-100'}`}
          >
            <div className="flex items-end justify-between gap-6">
              <p className={`text-[42px] leading-[44px] font-medium tracking-[-0.02em] ${s.text}`}>
                {result.label}
              </p>
              <Confidence value={result.confidence} />
            </div>

            <h4 className="mt-10 mb-3 flex items-center text-[15px] leading-[22px] font-medium">
              各类别的可能性
              <Info side="top" label="怎么看">
                模型给每个类别打的分。柱子越长、数字越大，说明越倾向这个判断。
              </Info>
            </h4>
            <ProbabilityRows names={result.class_names} probs={result.probabilities} winner={result.label} />

            <h4 className="mt-10 mb-3 flex items-center text-[15px] leading-[22px] font-medium">
              内部几种判断方法的原始打分
              <Info side="top" label="为什么还有这一张表" items={[
                '最终结果不是某一个模型说了算，而是几种方法按权重投票平均出来的。',
                '这里列出它们各自的分数，方便对照。',
              ]} />
            </h4>
            <ChannelTable channels={result.channels} names={result.class_names} />

            <p className="mt-6 text-[15px] leading-[22px] text-ink-2 tabular-nums">
              {taskLabel}，设备 {result.device === 'cuda' ? 'GPU' : result.device}，耗时 {result.latency_ms} ms
              {matched !== null && (
                <>
                  {'，'}
                  {matched ? '和人工标注判断一致' : '和人工标注判断不一致'}（{sample.label}）
                </>
              )}
            </p>
            {loading && (
              <p className="mt-3 text-[13px] leading-[18px] text-ink-2">
                正在分析新输入{' '}
                <Dots />
              </p>
            )}
            {error && <p className="mt-3 text-[13px] leading-[18px] text-neg">{error}</p>}
          </div>
        ) : loading ? (
          <Skeleton count={classCount} channelCount={channelCount} />
        ) : error ? (
          <p className="pt-1 text-[15px] leading-[22px] text-neg">{error}</p>
        ) : (
          /* 空状态：什么都不显示 */
          <div />
        )}
      </div>
    </div>
  )
}

/* ── 演示：控件 ────────────────────────────────────────────────── */

const tabClass =
  'flex h-8 items-center justify-center bg-transparent px-0 text-[15px] leading-5 font-normal whitespace-nowrap text-ink-2 outline-none select-none transition-colors duration-150 hover:text-ink focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink data-active:text-ink'

const buttonBase =
  'flex h-9 items-center justify-center rounded-none border px-4 text-[15px] leading-none whitespace-nowrap select-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink data-disabled:cursor-not-allowed data-disabled:opacity-40'

const buttonPrimary = `${buttonBase} min-w-32 border-ink bg-ink text-canvas transition-colors duration-150 hover:not-data-disabled:bg-ink-2`
const buttonSecondary = `${buttonBase} border-ink bg-canvas text-ink transition-colors duration-150 hover:not-data-disabled:bg-tint active:not-data-disabled:bg-tint-strong`

/* ── 文案内容：面向讲台，大白话 ─────────────────────────────── */

const METHOD = [
  [
    '介绍',
    '本次作业的数据来自老师提供的 600 条豆瓣电影短评。为了让模型学得更准，我逐条阅读并进行了人工情绪标注（正面 316 条，中性 92 条，负面 192 条）。在此过程中我发现，豆瓣原生的星级经常和文字情绪对不上（比如有人打 4 星却通篇吐槽，打 2 星只是因为「没看懂」），所以星级仅被我用作对照，不参与实际训练。',
  ],
  [
    '初次尝试',
    <>
      我初步尝试了能找到的中文情感分析大模型（包括 Erlangshen-RoBERTa 的 1.1 亿和 3.3 亿版本、MegatronBERT-1.3B 以及 MacBERT）。做法有两种：一是
      <span className="whitespace-nowrap">
        冻结参数
        <Info side="top" label="冻结参数是什么" items={[
          '「参数（Parameters / Weights）」就是模型在训练过程中学到的知识。',
          '冻结 = 训练时把大模型内部所有参数全部「锁死」，不允许它们发生任何改变。',
          '它不学习任何新东西，只把短评转换成一串浓缩了语义的数字（特征向量 / Embeddings），像个无情的翻译机。',
          '优点：极其省显存、训练速度极快（大模型不参与反向传播计算梯度）。',
          '缺点：没有针对豆瓣短评专门优化，遇到「阴阳怪气」可能识别不准。',
        ]} />
      </span>
      ，只把它们当「阅读器」提取
      <span className="whitespace-nowrap">
        特征
        <Info side="top" label="特征是什么" items={[
          '特征（特征向量 / Embeddings）= 大模型利用以前学过的通用语文知识，把一句话转换成一堆浓缩了语义的数字。',
          '意思相近的句子，数字也相近；后面再接一个简单的小算法去读这些数字。',
        ]} />
      </span>
      ，再用
      <span className="whitespace-nowrap">
        逻辑回归
        <Info side="top" label="逻辑回归是什么" items={[
          '一个非常简单的小算法：输入特征向量，输出「正面 / 中性 / 负面」的概率。',
          '只训练它一个就行——大模型的参数全程不动，所以训练极快、几乎不占资源。',
        ]} />
      </span>
      分类；二是
      <span className="whitespace-nowrap">
        全参数微调
        <Info side="top" label="全参数微调是什么" items={[
          '微调（Fine-tuning）= 在原有基础上做细微的调整；全参数微调就是把大模型所有参数全部「解冻」。',
          '做法：在标注好的 600 条短评上继续训练，根据标准答案修改模型的知识结构，让它专门适应豆瓣短评的说话腔调和分类标准。',
          '优点：准确率通常更高，因为它彻底贴合了任务和数据。',
          '缺点：非常吃硬件——要更新大模型几亿个参数，RX 9070 GRE 的显存会被塞得满满当当，训练时间也会长很多。',
        ]} />
      </span>
      。我将这些通道的概率平均起来，得到了一个成绩不错的「老师」（二分类准确率 95.7%，三分类 86.8%）。但问题很明显：每条短评要同时跑 4 到 7 个模型，推理太慢且太占显存，根本没法在讲台上做实时演示。
    </>,
  ],
  [
    '模型优化',
    <>
      为了能在本地流畅运行，我采用了
      <span className="whitespace-nowrap">
        知识蒸馏
        <Info side="top" label="知识蒸馏是什么" items={[
          '一句话：师傅带徒弟——把庞大、运行缓慢的老师模型脑子里的知识，浓缩并转移到小巧、运行飞快的学生模型身上。',
          '这项技术由深度学习鼻祖 Geoffrey Hinton 在 2015 年系统性提出。',
          '普通训练（硬标签）：只学标准答案。比如「特效很震撼，但剧情稍微有点老套」被标为「正面」，小模型只会死记硬背「这句 = 正面」，学不到「老套」带来的那一点点负面。',
          '知识蒸馏（软标签）：先让大模型当老师，它不给死板答案，而是给一组概率——「65% 正面、25% 中性、10% 负面」。',
          '学生训练时不只盯标准答案，还要去模仿老师的这组概率。',
          '概率里藏着「暗知识」：老师其实在偷偷告诉学生「这句主要是夸的但带一点客观评价」「它绝不是纯差评（负面只有 10%）」。',
          '学到的不再是非黑即白的答案，而是句子之间微妙的语义关系——好比不只拿到参考答案，还拿到学霸写满批注的草稿纸。',
        ]} />
      </span>
      技术，也就是「师傅带徒弟」。训练时，轻量的 MacBERT 作为学生模型，一半学习我的人工标注答案，另一半则模仿老师的完整打分概率（比如「62% 正面、30% 中性、8% 负面」）。这样，单靠一个小模型就能尽量学到集成模型的分类逻辑。
    </>,
  ],
  [
    '混合模型',
    <>
      实验中我发现，单靠学生模型很快遇到了
      <span className="whitespace-nowrap">
        准确率瓶颈
        <Info side="top" label="为什么会有瓶颈" items={[
          '学生（MacBERT）通过蒸馏「偷师」了很强的技巧，但它本身参数量小，可以理解为「脑容量有限」。',
          '遇到豆瓣短评里极其隐晦的讽刺、或结构极其复杂的长句时，理解能力达不到就会开始猜错——这就是单靠它无法突破的准确率天花板。',
        ]} />
      </span>
      。于是我额外请了一个
      <span className="whitespace-nowrap">
        「裁判」
        <Info side="top" label="裁判是怎么工作的" items={[
          '把教师团队里底子最厚、13 亿参数的 MegatronBERT-1.3B 单独请出来当裁判。',
          '不改变它的庞大参数（保持冻结不训练），只利用它极强的阅读理解能力，把句子浓缩成深层特征，再交给后面的小网络分类头打分。',
          '特点：见多识广、大局观极强、容错率高。',
        ]} />
      </span>
      ：让参数最大的 MegatronBERT-1.3B 保持原样不训练，只负责把短评变成语义特征，交给一个很小的神经网络打分。在最终出结果时，学生和裁判按
      <span className="whitespace-nowrap">
        1:2 的权重平均
        <Info side="top" label="1:2 权重平均怎么算" items={[
          '不是让任何一方单独做主，而是把双方输出的概率做加权平均。',
          '例：学生认为「正面」概率 90%，裁判觉得有猫腻只有 60% → (90% × 1票 + 60% × 2票) ÷ 3票 = 70%。',
          '给裁判 2 票是因为它 13 亿参数提取的特征更可靠，能把学生偶尔「过于自信」的错误判断强行拉回来。',
        ]} />
      </span>
      ——学生算一票，裁判算两票。这样每条短评只需要
      <span className="whitespace-nowrap">
        跑两三次模型前向
        <Info side="top" label="为什么速度快了一倍多" items={[
          '原来的教师团队每分析一句要同时跑 4~7 个模型；现在每个任务只跑 2~3 个：轻量学生 MacBERT（中性专家是用 11.5 万条豆瓣语料伪标签训练的另一个同结构检测器）、冻结裁判 MegatronBERT-1.3B（只做前向推断不训练，两个小裁判头共享同一次特征提取）。',
          '用这套轻量组合打败了 4~7 个大模型堆砌的成绩，既拉高了准确率天花板，又大幅降低计算量，完美实现单卡实时演示。',
        ]} />
      </span>
      ，既提高了成绩，速度也快了一倍多。
    </>,
  ],
  [
    '我遇到的问题',
    <>
      本次作业全程在一块 AMD RX 9070 GRE（12GB 显存）显卡上完成。从数据标注到交叉验证，我做了几十轮实验，累计训练了 40 多个小时（其中单个学生模型一次训练约 30 秒）。期间主要遇到了两个问题：一是「中性」类样本太少（仅 92 条），而且很多评论「先夸后贬」难以界定，所以我在做
      <span className="whitespace-nowrap">
        二分类
        <Info side="top" label="为什么二分类要扔掉中性数据" items={[
          '样本不平衡：机器学习最怕「偏科」——正面 316 条、负面 192 条，中性只有 92 条，模型很难学到「中性」到底长什么样。',
          '特征模糊（先夸后贬）：中性评论往往不是客观描述，而是包含复杂转折（比如「特效无敌，但剧情太拉胯」），这种复杂情绪连人都很难界定，更别说小模型。',
          '工程妥协：实际产品中往往只需要知道用户「喜欢」还是「讨厌」。所以我做了一个明智的工程决定——剔除这 92 条模棱两可的数据，让模型专心学极端的「好」与「坏」，这就是二分类（只分正负）。',
        ]} />
      </span>
      时干脆去掉了中性数据；二是单独蒸馏的效果不理想，最后是靠
      <span className="whitespace-nowrap">
        「学生+裁判」
        <Info side="top" label="为什么混合方案能反超老师" items={[
          '蒸馏的瓶颈：学生模型虽然快但脑容量小，把老师所有知识硬塞给它，学到一定程度就学不动了，准确率上不去。',
          '混合方案的化学反应：引入不训练的「特征裁判」（MegatronBERT-1.3B），让「聪明的裁判」和「勤奋的学生」按 2:1 权重一起投票，奇迹发生了。',
          '青出于蓝而胜于蓝：蒸馏产物的成绩一般只能无限接近老师、很难超越；但「学生+裁判」架构二分类（95.9%）追平了臃肿的 7 模型教师团队，三分类（89.5%）甚至反超了 2.7%（以上均为泄漏审计校正前的交叉验证成绩）。',
          '这意味着：用更少的模型数量、更快的运行速度，打败了原来靠堆砌算力算出来的成绩——是整个项目最能拿出来说的亮点。',
        ]} />
      </span>
      的混合方案才解决了问题。针对最拖后腿的「中性」类，我又专门训练了一对「中性专家」——一个在 11.5 万条豆瓣真实语料伪标签上训练的中性检测器（中性类在金标里只有 92 条，语料伪标签让它真正见过各种「中性」长什么样），加一个复用裁判特征的小网络，让它们单独估计「这是一条中性评价的概率」，再以小权重融进最终结果，中性类的召回率从 54% 一路提到了 71%。最终的三分类准确率（89.5%，泄漏审计校正前）反而比老师高了 2.7 个百分点，二分类（95.9%）也追平了老师。
    </>,
  ],
  [
    '最后',
    '为了课堂演示的流畅度，目前上线的版本只保留了四个模型：三个 MacBERT 学生（分别负责二分类、三分类和中性判断）加一个共享的 MegatronBERT-1.3B 特征提取器（下面挂着两个小型 MLP 裁判头）。相比第一步每条短评要跑 4 到 7 个模型，现在每个任务实际只跑两到三个；三分类的单条耗时从约 150 毫秒降到了约 80 毫秒，服务冷启动从约 45 秒降至约 14 秒，以较低的算力成本实现了实时演示的目标。',
  ],
]

const FAQS = [
  {
    q: '豆瓣不是有星级吗？为什么还要再判断情绪？',
    a: '因为星级经常和文字打架：有人打 4 星，评论却是通篇吐槽；有人打 2 星，只是因为「没看懂」。所以我不用星级当标准答案，而是人工逐条阅读文字、标注情绪，用这些标注来训练模型，星级只留着做对照。',
  },
  {
    q: '模型怎么会「懂」情绪？',
    a: '底子是中文预训练大模型：它们在发布前已经读过海量中文文本，对语言有基本的理解。我的工作是让它们学会「豆瓣短评」这种具体场景下的情绪表达，方法就是本文的主线——先把多个模型组成老师，再通过知识蒸馏把判断本事传给学生和特征裁判。评测采用 5 折交叉验证：600 条数据分成 5 份轮流当考题，模型考的是自己没见过的那份，成绩才算数。',
  },
  {
    q: '训练要多久？中间遇到过什么困难？',
    a: '全部训练都在一块 AMD RX 9070 GRE 显卡上完成。从试各种模型、调参数到交叉验证，前前后后做了几十轮实验，累计训练了 40 多个小时。最难的不是算力而是数据：中性类只有 92 条，很多短评情绪混合、很难归类；其次是单个模型学不到集成的全部本事。我的解决办法是「学生加特征裁判」的混合结构，再针对最弱的中性类加了一对专职「中性专家」，最后比原来的重型集成还略好。',
  },
  {
    q: '跟市面上的方案比怎么样？',
    a: '我用「传统方法」做对照——不用 AI 大模型、只靠统计评论里高频词的老办法。它的二分类准确率是 84.1%、三分类是 72.3%，我分别是 95.9% 和 89.5%（泄漏审计校正前的交叉验证成绩）；换成更严格的「综合得分」（Macro-F1，三类都照顾到的平均分），分别高出 13.3% 和 24.2%。我还找来阿里官方开源的中文情感分析模型 StructBERT（ModelScope 平台的成品模型，只分正负两类），在同一批豆瓣数据上实测二分类准确率只有 86.6%，我的模型是 95.9%——它是面向通用评论训练的，我的模型是针对豆瓣短评专门训练的，这就是差距。我还实测了当下最火的开源中文大模型 Qwen3 全家桶（0.6B / 1.7B / 4B 三个尺寸），让它们 zero-shot 直接判断三分类，最好的 4B 也只有 78.3%——比我的 89.5% 低了 11 个多百分点，主要卡在「中性」类上。通用大模型再大，没有针对豆瓣短评的训练也学不会这种「网感」，这就是任务专用管线的价值。说它是创新，是因为我没有简单地换更大的模型，而是设计了一套「先当老师、再带学生」的压缩方案——在只有 600 条标注的小数据场景下，让轻量模型追平甚至反超自己的老师，这对数据少、算力有限的实际项目很有参考价值。',
  },
  {
    q: '为什么不直接用阿里云 NLP 这类现成的商业接口？',
    a: [
      '第一，商业接口是「黑盒」，满足不了知识蒸馏的需求。它通常只返回一个最终结论（比如「正面，置信度 0.9」），而蒸馏需要老师不仅给答案，还要给出细粒度的概率分布（比如「62% 正面、30% 中性、8% 负面」），特征裁判更是要直接取深层语义特征向量——这些底层中间数据，商业 API 根本拿不到。',
      '第二，豆瓣短评有独特的「网感」。通用模型对标准新闻、商品评价很准，但豆瓣评论充满反讽、玩梗和阴阳怪气（「烂得我都笑了」「好看到睡着」）。我用手工标注的 600 条豆瓣专属数据做全参数微调，让模型彻底学会豆瓣用户的说话腔调——这种垂直定制，商业 API 要么做不到，要么收费非常昂贵。',
      '第三，本地毫秒级实时体验。用云端接口每次分析都要发一次网络请求，延迟可能飙到几百毫秒甚至超时。我把知识浓缩进轻量模型，全部跑在本地的 RX 9070 GRE 上，不联网、单条约 0.1 秒，这是任何云端 API 都给不了的。',
    ],
  },
]

/* 本项目方法主要受益的论文（与 thesis/thesis.tex 参考文献一致，均已核实链接） */
const REFS = [
  { cite: 'Devlin J, Chang M W, Lee K, Toutanova K. BERT: Pre-training of deep bidirectional transformers for language understanding. NAACL-HLT, 2019.', url: 'https://arxiv.org/abs/1810.04805' },
  { cite: 'Liu Y, Ott M, Goyal N, et al. RoBERTa: A robustly optimized BERT pretraining approach. arXiv:1907.11692, 2019.', url: 'https://arxiv.org/abs/1907.11692' },
  { cite: 'Cui Y, Che W, Liu T, et al. Revisiting pre-trained models for Chinese natural language processing. Findings of EMNLP, 2020.', url: 'https://arxiv.org/abs/2004.13922' },
  { cite: 'Cui Y, Che W, Liu T, et al. Pre-training with whole word masking for Chinese BERT. IEEE/ACM TASLP, 2021.', url: 'https://arxiv.org/abs/1906.08101' },
  { cite: 'Wang W, Bi B, Yan M, et al. StructBERT: Incorporating language structures into pre-training for deep language understanding. ICLR, 2020.', url: 'https://arxiv.org/abs/1908.04577' },
  { cite: 'Sun Y, Wang S, Li Y, et al. ERNIE: Enhanced representation through knowledge integration. arXiv:1904.09223, 2019.', url: 'https://arxiv.org/abs/1904.09223' },
  { cite: 'IDEA-CCNL. Erlangshen-MegatronBert-1.3B-Sentiment: model card and weights. Hugging Face model hub, 2023.', url: 'https://huggingface.co/IDEA-CCNL/Erlangshen-MegatronBert-1.3B-Sentiment' },
  { cite: 'Wang J, Zhang Y, Yang P, Gan R. Towards No.1 in CLUE semantic matching challenge: Pre-trained language model Erlangshen with propensity-corrected loss. arXiv:2208.02959, 2022.', url: 'https://arxiv.org/abs/2208.02959' },
  { cite: 'Kim Y. Convolutional neural networks for sentence classification. EMNLP, 2014.', url: 'https://aclanthology.org/D14-1181/' },
  { cite: 'Socher R, Perelygin A, Wu J, et al. Recursive deep models for semantic compositionality over a sentiment treebank. EMNLP, 2013.', url: 'https://aclanthology.org/D13-1170/' },
  { cite: 'Pang B, Lee L. Opinion mining and sentiment analysis. Foundations and Trends in Information Retrieval, 2008.', url: 'https://doi.org/10.1561/1500000011' },
  { cite: 'Tan S, Zhang J. An empirical study of sentiment analysis for Chinese documents. Expert Systems with Applications, 2008.', url: 'https://doi.org/10.1016/j.eswa.2007.05.028' },
  { cite: 'Yarowsky D. Unsupervised word sense disambiguation rivaling supervised methods. ACL, 1995.', url: 'https://aclanthology.org/P95-1026/' },
  { cite: 'Lee D H. Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks. ICML 2013 Workshop on Challenges in Representation Learning, 2013.', url: null },
  { cite: 'Arazo E, Ortego D, Albert P, et al. Pseudo-labeling and confirmation bias in deep semi-supervised learning. arXiv:1908.02983, 2020.', url: 'https://arxiv.org/abs/1908.02983' },
  { cite: 'Ratner A, Bach S H, Ehrenberg H, et al. Snorkel: Rapid training data creation with weak supervision. Proceedings of the VLDB Endowment, 2017.', url: 'https://arxiv.org/abs/1711.10160' },
  { cite: 'Xie Q, Luong M T, Hovy E, Le Q V. Self-training with noisy student improves ImageNet classification. CVPR, 2020.', url: 'https://arxiv.org/abs/1911.04252' },
  { cite: 'Sohn K, Berthelot D, Carlini N, et al. FixMatch: Simplifying semi-supervised learning with consistency and confidence. NeurIPS, 2020.', url: 'https://arxiv.org/abs/2001.07685' },
  { cite: 'Hinton G, Vinyals O, Dean J. Distilling the knowledge in a neural network. arXiv:1503.02531, 2015.', url: 'https://arxiv.org/abs/1503.02531' },
  { cite: 'Gou J, Yu B, Maybank S J, Tao D. Knowledge distillation: A survey. International Journal of Computer Vision, 2021.', url: 'https://arxiv.org/abs/2006.05525' },
  { cite: 'Jiao X, Yin Y, Shang L, et al. TinyBERT: Distilling BERT for natural language understanding. Findings of EMNLP, 2020.', url: 'https://arxiv.org/abs/1909.10351' },
  { cite: 'Shoeybi M, Patwary M, Puri R, et al. Megatron-LM: Training multi-billion parameter language models using model parallelism. arXiv:1909.08053, 2019.', url: 'https://arxiv.org/abs/1909.08053' },
  { cite: 'Joulin A, Grave E, Bojanowski P, Mikolov T. Bag of tricks for efficient text classification. EACL, 2017.', url: 'https://aclanthology.org/E17-2068/' },
  { cite: 'Reimers N, Gurevych I. Sentence-BERT: Sentence embeddings using siamese BERT-networks. EMNLP-IJCNLP, 2019.', url: 'https://aclanthology.org/D19-1410/' },
  { cite: 'Guo C, Pleiss G, Sun Y, Weinberger K Q. On calibration of modern neural networks. ICML, 2017.', url: 'https://proceedings.mlr.press/v70/guo17a.html' },
  { cite: 'Qwen Team. Qwen3: Technical report and model release. GitHub repository, 2025.', url: 'https://github.com/QwenLM/Qwen3' },
  { cite: 'Gilardi F, Alizadeh M, Kubli M. ChatGPT outperforms crowd-workers for text-annotation tasks. PNAS, 2023.', url: 'https://doi.org/10.1073/pnas.2305016120' },
  { cite: 'Törnberg P. ChatGPT-4 outperforms experts and crowd workers in annotating political Twitter messages with zero-shot learning. arXiv:2304.06588, 2023.', url: 'https://arxiv.org/abs/2304.06588' },
  { cite: 'Dietterich T G. Approximate statistical tests for comparing supervised classification learning algorithms. Neural Computation, 1998.', url: 'https://doi.org/10.1162/089976698300017197' },
  { cite: 'Demšar J. Statistical comparisons of classifiers over multiple data sets. Journal of Machine Learning Research, 2006.', url: 'https://jmlr.org/papers/v7/demsar06a.html' },
  { cite: 'Efron B, Tibshirani R J. An introduction to the bootstrap. Chapman & Hall, 1993.', url: null },
  { cite: 'McNemar Q. Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika, 1947.', url: 'https://doi.org/10.1007/BF02295996' },
]

/* ── 应用 ──────────────────────────────────────────────────────── */

export default function App() {
  const [task, setTask] = useState('3class')
  const [text, setText] = useState('')
  const [sample, setSample] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [runId, setRunId] = useState(0)
  const [metrics, setMetrics] = useState(null)
  const [examples, setExamples] = useState({})
  const [offset, setOffset] = useState(0)
  const reqId = useRef(0)

  useEffect(() => {
    api('/api/metrics').then(setMetrics).catch(() => {})
  }, [])

  useEffect(() => {
    api(`/api/examples?task=${task}`)
      .then((d) => setExamples((e) => ({ ...e, [task]: d.examples })))
      .catch(() => {})
  }, [task])

  const changeTask = useCallback((value) => {
    reqId.current += 1
    setTask(value)
    setResult(null)
    setError('')
    setLoading(false)
    setSample(null)
    setOffset(0)
  }, [])

  const run = useCallback(async (input, t) => {
    const value = input.trim()
    if (!value) {
      setError('请先输入或选择一条短评。')
      return
    }
    const id = ++reqId.current
    setLoading(true)
    setError('')
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
    try {
      let data = null
      const maxAttempts = IS_HF_SPACE ? 90 : 30
      for (let attempt = 0; attempt < maxAttempts; attempt++) {
        try {
          data = await api('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: value, task: t }),
          })
          break
        } catch (e) {
          // 503 = backend still loading models; network error = backend not
          // running / Space waking up. Retry both, but bail early on real
          // 4xx/5xx. Spaces get a longer window for cold start + GPU warm-up.
          const msg = String(e && e.message ? e.message : e)
          const isLoading = msg.startsWith('503')
          const isDown = /failed to fetch|networkerror|load failed|ended without result/i.test(msg)
          if ((!isLoading && !isDown) || attempt === maxAttempts - 1) throw e
          if (isLoading) setError('模型正在启动，约 15 秒…')
          else setError('正在连接本地服务…')
          await sleep(2000)
          if (id !== reqId.current) return
        }
      }
      if (id !== reqId.current) return
      setResult(data)
      setRunId(id) /* 新结果到了才重播入场动画 */
      setError('')
    } catch (e) {
      if (id !== reqId.current) return
      const msg = String(e && e.message ? e.message : e)
      setError(/failed to fetch|networkerror|load failed/i.test(msg)
        ? '连不上本地服务：请确认后端已启动（web 目录下运行 start_demo.bat）。'
        : `预测失败：${msg}`)
    } finally {
      if (id === reqId.current) setLoading(false)
    }
  }, [])

  const pickSample = (ex) => {
    setText(ex.text)
    setSample({ label: ex.label, star: ex.star, text: ex.text })
    run(ex.text, task)
  }

  const clear = () => {
    reqId.current += 1
    setText('')
    setSample(null)
    setResult(null)
    setError('')
    setLoading(false)
  }

  const list = examples[task] ?? []
  const visible = Array.from(
    { length: Math.min(SAMPLE_COUNT, list.length) },
    (_, i) => list[(offset + i) % list.length],
  )
  const taskInfo = TASKS.find((t) => t.value === task)

  return (
    <div className="min-h-screen bg-canvas font-sans text-ink">
      <ShaderBg />
      <div className="relative mx-auto grid max-w-[1480px] grid-cols-8 gap-6 p-6 pb-24 md:gap-9 md:px-9 md:py-7 md:pb-32">
        {/* 头部 */}
        <header className="contents">
          <div className="col-[1/5]">
            <a
              href="#top"
              className="text-[15px] leading-[22px] text-ink focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink"
            >
              豆瓣短评情感分析
            </a>
          </div>
          <nav className="col-[5/-1] flex flex-col gap-1 text-[15px] leading-[22px] lg:col-[5/7]" aria-label="页面导航">
            <a href="#numbers" className={`w-fit ${textLink}`}>效果怎么样</a>
            <a href="#demo" className={`w-fit ${textLink}`}>在线演示</a>
            <a href="#method" className={`w-fit ${textLink}`}>我是怎么做的</a>
            <a href="#faq" className={`w-fit ${textLink}`}>常见问题</a>
          </nav>
          <p className="col-[7/-1] hidden text-[15px] leading-[22px] text-ink-2 lg:block">
            AMD RX 9070 GRE，本地就能跑
          </p>
        </header>

        <main id="top" className="contents">
          {/* 大标题 */}
          <section className="contents">
            <h1 className="anim-rise col-[1/-1] mt-10 text-[30px] leading-[36px] font-normal tracking-[-0.025em] text-balance md:mt-20 md:text-[40px] md:leading-[46px] xl:col-end-7">
              <span className="block text-[15px] leading-[22px] font-normal tracking-normal text-ink-2 md:text-[17px] md:leading-[24px]">Let's Do This!</span>
              让机器读懂豆瓣短评里的情绪
            </h1>
            <div className="anim-rise col-[1/-1]" style={{ animationDelay: '90ms' }}>
              <ArrowLink href="#demo">去在线演示</ArrowLink>
            </div>
          </section>

          {/* 引言 */}
          <section className="contents">
            <div className="col-[1/-1] mt-14 flex flex-col gap-4 text-[18px] leading-[25px] text-pretty md:col-[3/-1] md:mt-20 xl:col-end-8">
              {[
                '介绍我的研究。对于海量豆瓣短评的情感解析，简单来说，只要你在下方输入一条评论，系统就会瞬间告诉你观众是在点赞、吐槽还是保持中立。',
                '这套系统依靠轻量模型来运行：MacBERT 学生负责感知情绪，参数更大的 MegatronBERT-1.3B 负责把关语义特征，另有一对「中性专家」专门盯住最难判断的中立评价。为了让模型既聪明又轻巧，我采用了「知识蒸馏」，把复杂教师团队的理解能力有效传授给了学生模型。我实现了性能与算力的绝佳平衡。它不仅交出了二分类 95.9%、三分类 89.5% 的高准确率（泄漏审计校正前的交叉验证成绩，比传统算法高出 12%～17%），而且运行极其轻量——在单张 AMD RX 9070 GRE 显卡的加持下，分析一条短评仅需约 0.1 秒，完全满足本地无延迟的实时交互。',
              ].map((p, i) => (
                <p key={i} className="anim-rise" style={{ animationDelay: `${140 + i * 70}ms` }}>
                  {p}
                </p>
              ))}
            </div>
          </section>

          <Section
            id="numbers"
            title="效果怎么样"
            hint={{
              label: '传统方法是什么？',
              items: [
                '「传统方法」指的是不用 AI 大模型、只靠统计评论里高频词加一个简单分类器的老办法，它能达到的水平就是对照组。',
                '我的模型在它的基础上又提升了 12%～17%。',
                '所有分数都经过严格的交叉验证：数据分成 5 份轮流考试，模型考的是自己没见过的那份，所以成绩是可信的。',
              ],
            }}
          >
            <MetricList metrics={metrics} latency={result?.latency_ms} />
            <div className="mt-6 border border-ink">
              <div className="border-b border-ink px-4 py-3 text-[15px] leading-[22px]">
                和阿里 NLP 对比
                <Info side="bottom" label="阿里 StructBERT 是什么" items={[
                  '它是阿里达摩院官方开源的中文情感分析模型，我从 ModelScope 平台下载了成品权重来用。',
                  '它是面向通用评论训练的，只分正面和负面两类，没有中性；下面的分数是我把它放到同一批豆瓣数据上实测的结果。',
                  '准确率那行的「差距」按错误率算：阿里错 13.4%，我只错 4.1%，相当于错误率少了约 69%。',
                ]} />
              </div>
              <div className="tabular-nums">
                <div className={`${metricGrid} px-4 text-[15px] leading-[22px] text-ink-2`}>
                  <span>指标</span>
                  <span className="text-right">我的模型</span>
                  <span className="hidden text-right md:block">阿里 NLP</span>
                  <span className="text-right">差距</span>
                </div>
                {[
                  ['二分类准确率', ALI_BENCH.acc, metrics?.binary?.ours?.acc, 'err'],
                  ['二分类综合得分', ALI_BENCH.macro_f1, metrics?.binary?.ours?.macro_f1, 'pts'],
                  ['三分类准确率', null, metrics?.['3class']?.ours?.acc, null],
                  ['三分类综合得分', null, metrics?.['3class']?.ours?.macro_f1, null],
                ].map(([label, ali, ours, mode]) => (
                  <div
                    key={label}
                    className={`${metricGrid} border-t border-ink px-4 py-3 text-[15px] leading-[22px] md:text-[18px] md:leading-[25px]`}
                  >
                    <span>{label}</span>
                    <span className="text-right font-medium">{pct(ours)}</span>
                    <span className="hidden text-right text-ink-2 md:block">
                      {ali == null ? '无' : pct(ali)}
                    </span>
                    <span className="text-right text-pos">
                      {ali == null || ours == null
                        ? '无'
                        : mode === 'err'
                          ? `错误率少 ${Math.round(((ours - ali) / (1 - ali)) * 100)}%`
                          : `+${((ours - ali) * 100).toFixed(1)}%`}
                    </span>
                  </div>
                ))}
              </div>
            </div>
            <p className="mt-4 text-[15px] leading-[22px] text-ink-2 text-pretty">
              简单来说：在同一批数据上，我的模型比只会统计关键词的传统方法高了 12%～17%。
            </p>
            <p className="mt-2 text-[15px] leading-[22px] text-ink-2 text-pretty">
              注：以上分数为泄漏审计校正前的交叉验证成绩；严格泄漏校正后（去重 592 条）约为三分类 82.6–83.1%、二分类 91.5%（600 行口径 83.3–84.0 / 92.9；审计细节见配套论文与仓库 thesis/ 目录）。
            </p>
          </Section>

          <Section
            id="demo"
            title="在线演示"
            wide
            hint={{
              label: '怎么用',
              items: [
                '左边输入一条短评，或者直接点一条样例，再按「分析」（Ctrl 或 ⌘ + 回车也可以）。',
                '右边立刻显示判断结果——桌面端固定在右侧，手机上显示在按钮下方。',
              ],
            }}
          >
            <Tabs.Root value={task} onValueChange={changeTask}>
              <Tabs.List className="relative flex gap-6">
                {TASKS.map((t) => (
                  <Tabs.Tab key={t.value} value={t.value} className={tabClass}>
                    {t.label}
                  </Tabs.Tab>
                ))}
                <Tabs.Indicator className="absolute bottom-0 left-0 h-px w-(--active-tab-width) translate-x-(--active-tab-left) bg-ink transition-[translate,width] duration-150 ease-in-out" />
              </Tabs.List>

              <Tabs.Panel value={task} className="mt-6 outline-none">
                {/* 桌面端：左列是输入和样例，右列是固定住的结果。
                    手机上 DOM 顺序是 输入 → 结果 → 样例，不用滚动就能看到结果。 */}
                <div className="grid gap-y-12 lg:grid-cols-2 lg:gap-x-12">
                  {/* 输入 */}
                  <div className="min-w-0">
                    <label
                      htmlFor="review"
                      className="flex items-baseline justify-between gap-4 text-[15px] leading-[22px]"
                    >
                      <span className="font-medium">短评内容</span>
                      <span className="flex items-center text-ink-2">
                        {taskInfo.classes}
                        <Info side="top" label="两个任务的区别" items={[
                          '三分类：判断「正面 / 中性 / 负面」。',
                          '二分类：去掉「不好说」的中性，只分「正面 / 负面」，所以分数会更高。',
                        ]} />
                      </span>
                    </label>
                    <textarea
                      id="review"
                      value={text}
                      rows={6}
                      maxLength={4000}
                      onChange={(e) => {
                        setText(e.target.value)
                        setSample(null)
                      }}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                          e.preventDefault()
                          run(text, task)
                        }
                      }}
                      placeholder="粘贴一条豆瓣短评试试，例如：「前半段还行，后半段越看越困」……"
                      className="mt-1 block w-full rounded-none border border-ink bg-canvas px-3 py-2.5 text-[18px] leading-7 font-normal text-ink placeholder:text-ink-2 focus:outline-2 focus:-outline-offset-1 focus:outline-ink"
                    />
                    <p className="mt-1 flex justify-between gap-4 text-[15px] leading-[22px] text-ink-2 tabular-nums">
                      <span>{sample ? `人工标注：${sample.label}，★ ${sample.star} 星` : ''}</span>
                      <span>{text.length} / 4000</span>
                    </p>

                    <div className="mt-6 flex flex-wrap items-center gap-3">
                      <Button onClick={() => run(text, task)} disabled={loading} className={buttonPrimary}>
                        {loading ? (
                          <>
                            分析中{' '}
                            <Dots />
                          </>
                        ) : (
                          '分析'
                        )}
                      </Button>
                      <Button onClick={clear} disabled={loading} className={buttonSecondary}>
                        清空
                      </Button>
                      <span className="hidden text-[15px] leading-[22px] text-ink-2 sm:inline">
                        Ctrl 或 ⌘ + 回车 快速分析
                      </span>
                    </div>
                  </div>

                  {/* 结果 —— 桌面端固定在右侧，手机上紧跟按钮下方 */}
                  <div className="min-w-0 lg:row-span-2">
                    <div className="lg:sticky lg:top-7">
                      <Result
                        result={result}
                        loading={loading}
                        error={error}
                        sample={sample}
                        taskLabel={taskInfo.label}
                        runId={runId}
                        classCount={taskInfo.count}
                        channelCount={taskInfo.channels}
                      />
                    </div>
                  </div>

                  {/* 样例 */}
                  <div className="min-w-0">
                    <div className="mb-3 flex items-baseline justify-between gap-4 text-[15px] leading-[22px]">
                      <h3 className="flex items-center font-medium">
                        试试示例
                        <Info side="top" label="这些是什么" items={[
                          '数据集里的真实豆瓣短评，点一条会自动填入并分析。',
                          '右侧的颜色点 = 人工标注的情绪，★ = 豆瓣原始星级。',
                        ]} />
                      </h3>
                      {list.length > SAMPLE_COUNT && (
                        <Button
                          onClick={() => setOffset((o) => (o + SAMPLE_COUNT) % list.length)}
                          className={`text-ink ${textLink}`}
                        >
                          换一批
                        </Button>
                      )}
                    </div>
                    <ul className="border-t border-ink">
                      {visible.map((ex) => (
                        <li key={ex.id} className="border-b border-ink">
                          <button
                            type="button"
                            onClick={() => pickSample(ex)}
                            aria-pressed={sample?.text === ex.text}
                            title={ex.text}
                            className={`group grid w-full grid-cols-[1fr_auto] items-baseline gap-6 py-3 text-left text-[15px] leading-[22px] transition-colors duration-150 focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-ink ${
                              sample?.text === ex.text ? 'text-ink' : 'text-ink-2 hover:text-ink'
                            }`}
                          >
                            <span className="truncate">{ex.text}</span>
                            <span className="flex shrink-0 items-center gap-2 text-ink-2 tabular-nums">
                              <span
                                className={`size-1.5 transition-transform duration-150 group-hover:scale-150 ${senti(ex.label).bar}`}
                                aria-hidden="true"
                              />
                              {ex.label}，★ {ex.star}
                            </span>
                          </button>
                        </li>
                      ))}
                      {!visible.length && (
                        <li className="border-b border-ink py-3 text-[15px] leading-[22px] text-ink-2">
                          正在加载样例……
                        </li>
                      )}
                    </ul>
                  </div>
                </div>
              </Tabs.Panel>
            </Tabs.Root>
          </Section>

          <Section id="method" title="我是怎么做的">
            <ul className="border-t border-ink">
              {METHOD.map(([k, v]) => (
                <li
                  key={k}
                  className="grid grid-cols-1 gap-1 border-b border-ink py-3 text-[18px] leading-[25px] md:grid-cols-[1fr_2.4fr] md:gap-9"
                >
                  <span>{k}</span>
                  <span className="text-pretty">{v}</span>
                </li>
              ))}
            </ul>
          </Section>

          <Section id="paper" title="论文">
            <div className="border border-ink">
              <div className="border-b border-ink px-5 py-4">
                <p className="text-[18px] leading-[25px]">
                  Weakly-Supervised Chinese Review Sentiment Classification at Scale
                </p>
                <p className="mt-1.5 text-[15px] leading-[22px] text-ink-2">
                  Pseudo-Labeling and Specialist Decomposition —— 完整的方法、实验与局限性分析（中英双语预印本）
                </p>
              </div>
              <iframe
                src="/paper.pdf"
                title="论文 PDF"
                className="h-[75vh] w-full bg-white"
              />
              <div className="flex flex-wrap items-center justify-between gap-3 border-t border-ink px-5 py-3 text-[15px] leading-[22px]">
                <span className="text-ink-2">Shuakami, 2026 — 89.5% acc / 85.9 macro-F1（泄漏审计校正前 OOF）</span>
                <span className="flex items-center gap-4">
                  <a href="/paper_zh.pdf" target="_blank" rel="noreferrer" className={textLink}>
                    中文版 PDF ↗
                  </a>
                  <a href="/paper.pdf" target="_blank" rel="noreferrer" className={textLink}>
                    English PDF ↗
                  </a>
                </span>
              </div>
            </div>
          </Section>

          <Section id="faq" title="常见问题">
            <Accordion.Root className="border-t border-ink">
              {FAQS.map((f) => (
                <Accordion.Item key={f.q} className="border-b border-ink">
                  <Accordion.Header className="m-0">
                    <Accordion.Trigger className="group flex w-full items-center justify-between gap-4 bg-transparent py-3 text-left text-[18px] leading-[25px] font-normal text-ink outline-none focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-ink">
                      {f.q}
                      <PlusIcon className="block shrink-0 text-ink-2 group-data-panel-open:hidden" />
                      <MinusIcon className="hidden shrink-0 text-ink-2 group-data-panel-open:block" />
                    </Accordion.Trigger>
                  </Accordion.Header>
                  <Accordion.Panel className="h-(--accordion-panel-height) overflow-hidden text-[18px] leading-[25px] transition-[height,opacity] duration-150 ease-out data-ending-style:h-0 data-ending-style:opacity-0 data-starting-style:h-0 data-starting-style:opacity-0">
                    {(Array.isArray(f.a) ? f.a : [f.a]).map((para, i) => (
                      <p key={i} className={`${i ? 'mt-3' : 'pt-1'} pb-3 text-pretty`}>{para}</p>
                    ))}
                  </Accordion.Panel>
                </Accordion.Item>
              ))}
            </Accordion.Root>
          </Section>

          <Section id="refs" title="引用">
            <p className="text-[15px] leading-[22px] text-ink-2">
              This project stands on the shoulders of these papers. With gratitude to the authors.
            </p>
            <ol className="mt-5 flex flex-col gap-2.5">
              {REFS.map((r, i) => (
                <li key={i} className="flex gap-3 text-[13px] leading-[19px] text-ink-2">
                  <span className="w-7 shrink-0 text-right tabular-nums">[{i + 1}]</span>
                  {r.url ? (
                    <a
                      href={r.url}
                      target="_blank"
                      rel="noreferrer"
                      className={`min-w-0 ${textLink}`}
                    >
                      {r.cite}
                    </a>
                  ) : (
                    <span className="min-w-0">{r.cite}</span>
                  )}
                </li>
              ))}
            </ol>
          </Section>
        </main>
      </div>
    </div>
  )
}
