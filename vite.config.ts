import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// PORT comes from the untracked .envrc; the fallback keeps `npm run dev` working
// from a shell that hasn't loaded it.
const port = Number(process.env.PORT) || 3895;

export default defineConfig({
  plugins: [react()],
  // Only this machine by default. `npm run dev -- --host` opens it to the network, and then
  // anyone there can fetch any file of the repo the dev server reads, so it also refuses the
  // environment file and the training folder (ml/data holds the reference video's frames).
  server: {
    port,
    fs: { deny: [".env", ".env.*", ".envrc*", "*.{crt,pem,key,p12,pfx,cer,der}", ".npmrc", ".yarnrc.yml", "**/.git/**", fileURLToPath(new URL("ml/**", import.meta.url))] },
  },
  // sheet.html is a second page (the contact sheet of a move); it needs its own entry in the build.
  build: {
    rollupOptions: { input: { main: "index.html", sheet: "sheet.html" } },
  },
  preview: { port },
});
