import { afterEach, describe, expect, it, vi } from "vitest";
import { decimal, dict, getLang, resolveLang, setLang, t } from "../src/i18n/lang";
import { statusOf } from "../src/sim/status";
import type { LiveGinga } from "../src/sim/simulation";

describe("language choice", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("opens in English on a first visit, whatever the browser speaks", () => {
    expect(resolveLang("", null)).toBe("en");
    expect(resolveLang("?x=1", "fr")).toBe("en");
  });

  it("opens in Portuguese when the URL or the stored choice says so", () => {
    expect(resolveLang("?lang=pt", null)).toBe("pt");
    expect(resolveLang("", "pt")).toBe("pt");
  });

  it("lets the URL win over the stored choice", () => {
    expect(resolveLang("?lang=en", "pt")).toBe("en");
    expect(resolveLang("?lang=pt", "en")).toBe("pt");
    expect(resolveLang("?lang=xx", "pt")).toBe("pt");
  });

  it("remembers the choice and writes decimals the way the language does", () => {
    const saved = new Map<string, string>();
    vi.stubGlobal("localStorage", { getItem: (k: string) => saved.get(k) ?? null, setItem: (k: string, v: string) => saved.set(k, v) });
    setLang("pt");
    expect([...saved.values()]).toEqual(["pt"]);
    expect(getLang()).toBe("pt");
    expect(t("stepTitle", { dt: decimal(0.03, 2) })).toBe("Um passo de controle, 0,03 s (.)");
    setLang("en");
    expect(t("stepTitle", { dt: decimal(0.03, 2) })).toBe("One control step, 0.03 s (.)");
    expect(dict().sheetViews.front).toBe("front");
  });
});

/** The parts of a LiveGinga the status line reads. */
const ginga = (over: Partial<Record<string, unknown>>) =>
  ({
    phase: 0.25,
    period: 2.4,
    cycles: 2,
    bpm: 66,
    side: "r",
    feintMode: false,
    move: null,
    maneuver: null,
    pending: null,
    recover: null,
    fallen: false,
    files: { reference: { floor: { fallTime: 1 } } },
    ...over,
  }) as unknown as LiveGinga;

const lines = (g: LiveGinga) => {
  setLang("en");
  const en = statusOf(g).label;
  setLang("pt");
  const pt = statusOf(g).label;
  setLang("en");
  return { en, pt };
};

describe("status line", () => {
  it("reads the ginga cycle", () => {
    expect(lines(ginga({}))).toEqual({
      en: "real-time physics · cycle 3 · 66 bpm",
      pt: "física em tempo real · ciclo 3 · 66 bpm",
    });
  });

  it("says what a kick waits for, with the move name in Portuguese in both", () => {
    expect(lines(ginga({ pending: { move: "martelo", side: "l" } }))).toEqual({
      en: "Martelo · left leg · waiting for the moment",
      pt: "Martelo · perna esquerda · esperando o momento",
    });
  });

  it("names the kick in flight and which way an esquiva goes", () => {
    expect(lines(ginga({ move: "bencao", side: "r" }))).toEqual({ en: "Bênção · right leg", pt: "Bênção · perna direita" });
    expect(lines(ginga({ move: "esquiva", side: "r" }))).toEqual({ en: "Esquiva lateral · to the left", pt: "Esquiva lateral · para a esquerda" });
    expect(lines(ginga({ move: "martelo", feintMode: true }))).toEqual({
      en: "Martelo feint · right leg",
      pt: "Finta de Martelo · perna direita",
    });
  });

  it("follows a fall, the get-up and a restart", () => {
    expect(lines(ginga({ recover: { move: "joelhos", t: 0.2 } }))).toEqual({ en: "falling · hands and knees", pt: "caindo · mãos e joelhos" });
    expect(lines(ginga({ recover: { move: "quatro", t: 1.5 } }))).toEqual({
      en: "getting up · from the queda de quatro",
      pt: "levantando · da queda de quatro",
    });
    expect(lines(ginga({ fallen: true }))).toEqual({
      en: "real-time physics · fell — restarting",
      pt: "física em tempo real · caiu — recomeçando",
    });
  });

  it("names a maneuver in Portuguese", () => {
    expect(lines(ginga({ maneuver: { type: "role" } })).en).toBe("Rolê");
  });
});
