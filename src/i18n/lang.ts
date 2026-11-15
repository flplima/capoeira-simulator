import { useSyncExternalStore } from "react";
import { en, type Dictionary } from "./en";
import { pt } from "./pt";

export type Lang = "en" | "pt";
export const LANGS: Lang[] = ["en", "pt"];

const STORAGE_KEY = "capoeira.lang";
const dictionaries: Record<Lang, Dictionary> = { en, pt };
const isLang = (value: string | null | undefined): value is Lang => value === "en" || value === "pt";

/** The URL's `?lang=` wins, then the stored choice; everyone else gets English, whatever the browser says. */
export function resolveLang(search: string, stored: string | null): Lang {
  const fromUrl = new URLSearchParams(search).get("lang");
  if (isLang(fromUrl)) return fromUrl;
  return isLang(stored) ? stored : "en";
}

const storage = {
  get: () => {
    try {
      return localStorage.getItem(STORAGE_KEY);
    } catch {
      return null;
    }
  },
  set: (lang: Lang) => {
    try {
      localStorage.setItem(STORAGE_KEY, lang);
    } catch {
      // Private mode: the choice lasts until the page closes.
    }
  },
};

const listeners = new Set<() => void>();
let current: Lang = "en";

function apply(lang: Lang) {
  current = lang;
  if (typeof document !== "undefined") document.documentElement.lang = lang === "pt" ? "pt-BR" : "en";
}

if (typeof location !== "undefined") {
  apply(resolveLang(location.search, storage.get()));
  if (isLang(new URLSearchParams(location.search).get("lang"))) storage.set(current);
}

export const getLang = () => current;
export const subscribeLang = (listener: () => void) => {
  listeners.add(listener);
  return () => void listeners.delete(listener);
};

export function setLang(lang: Lang) {
  storage.set(lang);
  if (lang === current) return;
  apply(lang);
  for (const listener of listeners) listener();
}

/** Re-renders the component when the language changes. */
export const useLang = () => useSyncExternalStore(subscribeLang, getLang);

/** The current language's dictionary, for the maps keyed by move, maneuver or toque. */
export const dict = (): Dictionary => dictionaries[current];

export type StringKey = { [K in keyof Dictionary]: Dictionary[K] extends string ? K : never }[keyof Dictionary];

/** A string of the current language; `{name}` placeholders take their value from `vars`. */
export function t(key: StringKey, vars: Record<string, string | number> = {}): string {
  return dict()[key].replace(/\{(\w+)\}/g, (_, name: string) => String(vars[name] ?? `{${name}}`));
}

/** A number written the way the language writes it (0.03 s, 0,03 s). */
export function decimal(value: number, digits?: number): string {
  const text = digits === undefined ? String(value) : value.toFixed(digits);
  return current === "pt" ? text.replace(".", ",") : text;
}
