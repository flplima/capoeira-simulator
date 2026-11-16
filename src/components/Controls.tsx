import { useSyncExternalStore } from "react";
import { CapoeiraContext } from "../context";
import type { Playback } from "../playback";
import { MANEUVER_NAMES, MOVE_NAMES } from "../sim/types";
import { ask, OFFERED, type Offered } from "../offered";
import { SPEEDS } from "../machine";
import { decimal, dict, t, useLang } from "../i18n/lang";
import { berimbau } from "../music/berimbau";
import { DEFAULT_TOQUE, MAX_BPM, MIN_BPM, TOQUES } from "../music/toques";

const BPM_STEP = 2;

/** The button of one offered move or maneuver; `number` is its number key. */
function OfferButton({ playback, offered, ready, number }: { playback: Playback; offered: Offered; ready: boolean; number: number }) {
  const send = CapoeiraContext.useActorRef().send;
  const move = offered.kind === "move";
  const name = move ? MOVE_NAMES[offered.id] : MANEUVER_NAMES[offered.id];
  const hint = move ? dict().moveHints[offered.id] : dict().maneuverHints[offered.id];
  return (
    <button
      disabled={!ready || (move && !playback.sim?.available(offered.id))}
      onClick={() => {
        if (playback.sim) ask(playback.sim, offered);
        send({ type: "SHOW", topic: offered.topic });
      }}
      title={t(move ? "moveTitle" : "maneuverTitle", { name, hint, key: number })}
    >
      {name}
    </button>
  );
}

export function Controls({ playback }: { playback: Playback }) {
  useLang();
  const actor = CapoeiraContext.useActorRef();
  const speed = CapoeiraContext.useSelector((s) => s.context.speed);
  const playing = CapoeiraContext.useSelector((s) => s.matches("playing"));
  const live = useSyncExternalStore(playback.subscribe, () => playback.liveMode);
  // A move can be asked for while the body is gingando; the buttons wait for it to come back to the ginga.
  const ready = live === "ginga" && playing;
  const music = CapoeiraContext.useSelector((s) => s.context.music);
  const toqueId = CapoeiraContext.useSelector((s) => s.context.toque);
  const bpm = CapoeiraContext.useSelector((s) => s.context.bpm);
  const toque = TOQUES.find((q) => q.id === toqueId) ?? DEFAULT_TOQUE;
  const send = actor.send;

  return (
    <div className="controls">
      <div className="player" role="group" aria-label={t("playerGroup")}>
        <div className="transport">
          <button className="play" onClick={() => send({ type: "TOGGLE_PLAY" })} aria-label={t(playing ? "pause" : "play")} title={t("playPauseTitle")}>
            {playing ? "❚❚" : "▶"}
          </button>
          <button onClick={() => send({ type: "STEP" })} disabled={!playback.sim} aria-label={t("step")} title={t("stepTitle", { dt: decimal(0.03, 2) })}>
            ⏭
          </button>
        </div>

        <div className="segmented" role="group" aria-label={t("speed")}>
          {SPEEDS.map((s) => (
            <button key={s} aria-pressed={s === speed} onClick={() => send({ type: "SET_SPEED", speed: s })}>
              {decimal(s)}×
            </button>
          ))}
        </div>
        <button
          className="music-toggle"
          aria-pressed={music}
          onClick={() => {
            // Audio may only start inside a user gesture.
            berimbau.unlock();
            send({ type: "TOGGLE_MUSIC" });
          }}
          title={t("berimbauTitle")}
        >
          {t("berimbauToggle")}
        </button>
        <label className="select toque" title={dict().toqueDescriptions[toque.id]}>
          <span className="sr-only">{t("toque")}</span>
          <select value={toque.id} onChange={(e) => send({ type: "SET_TOQUE", id: e.target.value })}>
            {TOQUES.map((q) => (
              <option key={q.id} value={q.id} title={dict().toqueDescriptions[q.id]}>
                {q.name} · {q.bpm} bpm
              </option>
            ))}
          </select>
        </label>
        <div className="segmented tempo" role="group" aria-label={t("tempoGroup")}>
          <button onClick={() => send({ type: "SET_BPM", bpm: bpm - BPM_STEP })} disabled={bpm <= MIN_BPM} aria-label={t("slower")} title={t("slower")}>
            −
          </button>
          <output aria-live="polite" title={t("tempoTitle")}>
            {bpm} bpm
          </output>
          <button onClick={() => send({ type: "SET_BPM", bpm: bpm + BPM_STEP })} disabled={bpm >= MAX_BPM} aria-label={t("faster")} title={t("faster")}>
            +
          </button>
        </div>
      </div>
      <div className="moves" role="group" aria-label={t("movesGroup")}>
        {OFFERED.map((offered, i) => (
          <OfferButton key={offered.id} {...{ playback, offered, ready, number: i + 1 }} />
        ))}
      </div>
    </div>
  );
}
