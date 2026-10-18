import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// PORT comes from the untracked .envrc; the fallback keeps `npm run dev` working
// from a shell that hasn't loaded it.
const port = Number(process.env.PORT) || 3895;

export default defineConfig({
  plugins: [react()],
  server: { port, host: true },
  preview: { port, host: true },
});
