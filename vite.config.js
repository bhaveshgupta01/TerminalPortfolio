import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// Serves the functions in api/ during `npm run dev`, so the chat works locally without
// the Vercel CLI. In production Vercel runs the same file as a function.
const devApi = () => ({
  name: 'dev-api',
  configureServer(server) {
    // Make ANTHROPIC_API_KEY etc. from .env.local visible to the handler.
    Object.assign(process.env, loadEnv(server.config.mode, process.cwd(), ''))

    server.middlewares.use('/api', async (req, res, next) => {
      const name = /^\/([a-z]+)(?:\?|$)/.exec(req.url)?.[1]
      if (!name) return next()
      try {
        const chunks = []
        for await (const chunk of req) chunks.push(chunk)
        const mod = await server.ssrLoadModule(`/api/${name}.js`)
        const handler = mod[req.method]
        if (!handler) {
          res.statusCode = 405
          res.end()
          return
        }
        const response = await handler(
          new Request(`http://localhost${req.originalUrl}`, {
            method: req.method,
            headers: req.headers,
            body: chunks.length && req.method !== 'GET' ? Buffer.concat(chunks) : undefined,
          }),
        )
        res.statusCode = response.status
        response.headers.forEach((value, key) => res.setHeader(key, value))
        res.end(await response.text())
      } catch (err) {
        console.error('[dev-api]', err)
        res.statusCode = 500
        res.setHeader('content-type', 'application/json')
        res.end(JSON.stringify({ error: 'unavailable' }))
      }
    })
  },
})

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), devApi()],
})
