import { parse } from "yaml";
import { z } from "zod";
import { MovementTextSchema, type MovementText } from "./schema";
import type { Lang } from "../i18n/lang";
import type { ReadText } from "./load";

export type { MovementText };

/** The movement's texts in a language; the names and sources are the same in both. */
export function textIn(movement: MovementText, lang: Lang) {
  const { name, aka, sources } = movement;
  const { translation, category, summary, steps, tips } = lang === "en" && movement.en ? movement.en : movement;
  return { name, aka, sources, translation, category, summary, steps, tips };
}

/** The text of some movement files (name, how to do it, tips, sources), without compiling their keyframes. */
export async function loadTexts<Id extends string>(read: ReadText, ids: Id[]): Promise<Partial<Record<Id, MovementText>>> {
  const texts: Partial<Record<Id, MovementText>> = {};
  await Promise.all(
    ids.map(async (id) => {
      const path = `movements/${id}.yaml`;
      const result = MovementTextSchema.safeParse(parse(await read(path)));
      if (!result.success) throw new Error(`${path}: ${z.prettifyError(result.error)}`);
      texts[id] = result.data;
    }),
  );
  return texts;
}
