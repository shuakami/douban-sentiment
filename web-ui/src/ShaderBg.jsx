import { useEffect, useRef, useState } from 'react'

const FADE_PX = 640

const PALETTES = {
  light: {
    colors: ['#f0f5ff', '#d9e6ff', '#317cfe14'],
    intensity: 0.4,
  },
  dark: {
    colors: ['#0a0f1e', '#0e1a33', '#1d4ed840'],
    intensity: 0.55,
  },
}

function scrollFade() {
  return Math.max(0, Math.min(1, 1 - window.scrollY / FADE_PX))
}

/* paper-design/shaders 的 grainGradient 渐变背景：
   外层 fixed 铺满视口、随滚动渐隐；内层只播一次入场动画
   （淡入 + 轻微缩放，expo 缓出贝塞尔，见 index.css 的 .anim-shader-in）。
   浅色用浅蓝、深色用深夜蓝，随系统主题切换重挂载。
   开启「减弱动效」时不挂载。 */
export default function ShaderBg() {
  const hostRef = useRef(null)
  const [scheme, setScheme] = useState(() =>
    window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light',
  )
  const [scrollO, setScrollO] = useState(0)
  const [entered, setEntered] = useState(false)

  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    const onChange = (e) => setScheme(e.matches ? 'dark' : 'light')
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [])

  useEffect(() => {
    setEntered(false)
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      return undefined
    }
    const host = hostRef.current
    if (!host) return undefined
    const pal = PALETTES[scheme]

    let mount = null
    let disposed = false
    let alive = true

    ;(async () => {
      try {
        const m = await import('@paper-design/shaders')
        if (disposed || !alive || !hostRef.current) return
        const s = m.defaultObjectSizing
        const tex = m.getShaderNoiseTexture()
        if (tex && !(tex.complete && tex.naturalWidth)) {
          await new Promise((r) => {
            tex.onload = r
            tex.onerror = r
          })
        }
        if (disposed || !alive || !hostRef.current) return
        mount = new m.ShaderMount(
          hostRef.current,
          m.grainGradientFragmentShader,
          {
            u_colorBack: m.getShaderColorFromString('#00000000'),
            u_colors: pal.colors.map(m.getShaderColorFromString),
            u_colorsCount: 3,
            u_softness: 1,
            u_intensity: pal.intensity,
            u_noise: 0.1,
            u_shape: m.GrainGradientShapes.corners,
            u_noiseTexture: tex,
            u_fit: m.ShaderFitOptions[s.fit],
            u_scale: s.scale,
            u_rotation: s.rotation,
            u_offsetX: s.offsetX,
            u_offsetY: s.offsetY,
            u_originX: s.originX,
            u_originY: s.originY,
            u_worldWidth: s.worldWidth,
            u_worldHeight: s.worldHeight,
          },
          undefined,
          0.4,
          0,
          1,
          1920 * 1080,
        )
        // 双 rAF 确保 shader 先渲染一帧，再播入场动画，避免黑闪
        requestAnimationFrame(() =>
          requestAnimationFrame(() => {
            if (alive) setEntered(true)
          }),
        )
      } catch {
        /* 着色器加载失败时静默降级为纯色背景 */
      }
    })()

    const onScroll = () => setScrollO(scrollFade())
    window.addEventListener('scroll', onScroll, { passive: true })
    onScroll()

    return () => {
      alive = false
      disposed = true
      window.removeEventListener('scroll', onScroll)
      if (mount && typeof mount.dispose === 'function') mount.dispose()
      setScrollO(0)
      setEntered(false)
    }
  }, [scheme])

  return (
    <div
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 z-0 transition-opacity duration-150 ease-out"
      style={{ opacity: scrollO }}
    >
      {/* key=scheme：主题切换时重建节点，入场动画重播 */}
      <div
        key={scheme}
        ref={hostRef}
        className={`absolute inset-0 ${entered ? 'anim-shader-in' : 'opacity-0'}`}
      />
    </div>
  )
}
