# web-ui

React + Vite frontend for the Douban sentiment demo. Built with Base UI
primitives and Tailwind-style utility classes; includes a grain-gradient
shader background and an embedded paper viewer.

```bash
npm install
npm run dev      # dev server, proxies /api to the FastAPI backend on :8000
npm run build    # emits the production bundle to ../web/static
```

The backend (`../web/app.py`) serves `../web/static` directly, so a fresh
`npm run build` is all that is needed before running the demo.
