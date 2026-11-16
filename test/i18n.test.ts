import { afterEach, describe, expect, it, vi } from "vitest";
import { decimal, dict, getLang, resolveLang, setLang, t } from "../src/i18n/lang";

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
