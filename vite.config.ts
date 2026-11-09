import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// PORT comes from the untracked .envrc; the fallback keeps `npm run dev` working
// from a shell that hasn't loaded it.
const port = Number(process.env.PORT) || 3895;

export default defineConfig({
  plugins: [react()],
  server: { port, host: true },
  // Deployed, `vite preview` serves dist/ behind the edge proxy, which arrives with the public Host
  // header. It binds only to 127.0.0.1, so accepting any Host opens nothing; it just avoids Vite's
  // host check blocking the site.
  preview: { port, host: "127.0.0.1", allowedHosts: true },
});
