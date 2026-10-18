import { parse } from "yaml";
import { z } from "zod";
import { compileMovement, PoseResolver, type Clip } from "./compile";
import { bakeDynamics } from "../physics/bake";
import { IndexSchema, MovementSchema, PoseLibrarySchema, type Movement } from "./schema";

export interface Library {
  movements: Movement[];
  clips: Record<string, Clip>;
  /** Files that failed to parse, validate or compile; the rest still load. */
  errors: string[];
}

export type ReadText = (path: string) => Promise<string>;

function parseFile<T extends z.ZodType>(schema: T, text: string, path: string): z.infer<T> {
  const result = schema.safeParse(parse(text));
  if (!result.success) throw new Error(`${path}: ${z.prettifyError(result.error)}`);
  return result.data;
}

/**
 * Loads index.yaml, poses.yaml and every movement file it lists, validates
 * them and compiles each movement into a clip. A broken movement file is
 * reported in `errors` instead of taking the whole library down. `read`
 * resolves a path relative to the data directory — fetch in the browser, fs in
 * tests. `only` limits loading to some ids (the contact sheet loads one).
 */
export async function loadLibrary(read: ReadText, only?: string[]): Promise<Library> {
  const [index, poses] = await Promise.all([
    read("index.yaml").then((t) => parseFile(IndexSchema, t, "index.yaml")),
    read("poses.yaml").then((t) => parseFile(PoseLibrarySchema, t, "poses.yaml")),
  ]);
  const resolver = new PoseResolver(poses);
  const ids = index.movements.filter((id) => !only || only.includes(id));
  const results = await Promise.allSettled(
    ids.map(async (id) => {
      const path = `movements/${id}.yaml`;
      try {
        const movement = parseFile(MovementSchema, await read(path), path);
        if (movement.id !== id) throw new Error(`id "${movement.id}" does not match its file name`);
        return { movement, clip: bakeDynamics(compileMovement(movement, resolver)) };
      } catch (e) {
        const message = (e as Error).message;
        throw new Error(message.startsWith(path) ? message : `${path}: ${message}`);
      }
    }),
  );
  const library: Library = { movements: [], clips: {}, errors: [] };
  for (const r of results) {
    if (r.status === "rejected") library.errors.push((r.reason as Error).message);
    else {
      library.movements.push(r.value.movement);
      library.clips[r.value.movement.id] = r.value.clip;
    }
  }
  return library;
}

export const fetchText: ReadText = async (path) => {
  const res = await fetch(`${import.meta.env.BASE_URL}data/${path}`);
  // Dev servers answer a missing file with the HTML app shell, not a 404.
  if (!res.ok || res.headers.get("content-type")?.includes("text/html")) throw new Error(`${path}: não encontrado`);
  return res.text();
};
