import { describe, expect, it } from "vitest";
import { TOPICS } from "../src/machine";
import { OFFERED, offeredForKey } from "../src/offered";
import { DEFAULT_TOQUE, MAX_BPM, MIN_BPM, TOQUES } from "../src/music/toques";

describe("what the app offers", () => {
  it("is five moves and the giro de base, in that order", () => {
    expect(OFFERED.map((o) => o.id)).toEqual(["meia-lua-de-frente", "joelhada", "bencao", "esquiva", "cocorinha", "giro-de-base"]);
  });

  it("gives each its info tab, after the ginga", () => {
    expect(TOPICS).toEqual(["ginga", "meia-lua-de-frente", "joelhada", "bencao", "esquiva-lateral", "cocorinha", "giro-de-base"]);
  });

  it("answers the number keys in the order of the buttons, and nothing past the last", () => {
    OFFERED.forEach((o, i) => expect(offeredForKey(String(i + 1))).toBe(o));
    expect(offeredForKey(String(OFFERED.length + 1))).toBeUndefined();
    expect(offeredForKey("0")).toBeUndefined();
    expect(offeredForKey("e")).toBeUndefined();
  });

  it("opens on São Bento Pequeno at 90 bpm, with every toque inside the tempo range", () => {
    expect(DEFAULT_TOQUE.name).toBe("São Bento Pequeno");
    expect(DEFAULT_TOQUE.bpm).toBe(90);
    for (const toque of TOQUES) {
      expect(toque.bpm).toBeGreaterThanOrEqual(MIN_BPM);
      expect(toque.bpm).toBeLessThanOrEqual(MAX_BPM);
    }
  });
});
