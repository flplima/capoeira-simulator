import { useSyncExternalStore } from "react";
import { CapoeiraContext } from "../context";
import type { Playback } from "../playback";
import {
  MANEUVER_NAMES,
  MANEUVERS,
  MOVE_NAMES,
  MOVES,
  type GingaStyle,
  type Move,
} from "../sim/types";
import { SPEEDS, type Topic } from "../machine";
import { decimal, dict, t, useLang, type StringKey } from "../i18n/lang";
import { berimbau } from "../music/berimbau";
import { MAX_BPM, MIN_BPM, TOQUES } from "../music/toques";

const BPM_STEP = 2;

/** The info panel's entry for each move. */
export const MOVE_TOPICS: Record<Move, Topic> = {
  martelo: "martelo",
  bencao: "bencao",
  "meia-lua-de-frente": "meia-lua-de-frente",
  queixada: "queixada",
  ponteira: "ponteira",
  joelhada: "joelhada",
  "meia-lua-de-compasso": "meia-lua-de-compasso",
  esquiva: "esquiva-lateral",
  "esquiva-baixa": "esquiva-baixa",
  cocorinha: "cocorinha",
};

const MODE_LABELS = {
  loading: "modeLoading",
  failed: "modeFailed",
  waiting: "modeWaiting",
  kick: "modeKick",
  down: "modeDown",
  ginga: "modeGinga",
} as const;

const STYLE_LABELS: Record<GingaStyle, StringKey> = {
  auto: "styleAuto",
  angola: "styleAngola",
  padrao: "stylePadrao",
  regional: "styleRegional",
};

export function Controls({ playback }: { playback: Playback }) {
  useLang();
  const actor = CapoeiraContext.useActorRef();
  const speed = CapoeiraContext.useSelector((s) => s.context.speed);
  const playing = CapoeiraContext.useSelector((s) => s.matches("playing"));
  const live = useSyncExternalStore(playback.subscribe, () => playback.liveMode);
  const feintGinga = useSyncExternalStore(playback.subscribe, () => playback.sim?.pendingFeintGinga ?? false);
  const feintMode = useSyncExternalStore(playback.subscribe, () => playback.sim?.feintMode ?? false);
  const orbit = useSyncExternalStore(playback.subscribe, () => playback.sim?.nav.orbit ?? false);
  const gingaStyle = useSyncExternalStore(playback.subscribe, () => playback.sim?.gingaStyle ?? "auto");
  // A move can be asked for while the body is gingando; the buttons wait for it to come back to the ginga.
  const ready = live === "ginga" && playing;
  const music = CapoeiraContext.useSelector((s) => s.context.music);
  const toqueId = CapoeiraContext.useSelector((s) => s.context.toque);
  const bpm = CapoeiraContext.useSelector((s) => s.context.bpm);
  const toque = TOQUES.find((q) => q.id === toqueId) ?? TOQUES[0];
  const send = actor.send;

  return (
    <div className="controls">
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

      <div className="moves" role="group" aria-label={t("movesGroup")}>
        <span className="moves-label">{t(MODE_LABELS[live])}</span>
        {MOVES.map((move, i) => (
          <button
            key={move}
            disabled={!ready || !playback.sim?.available(move)}
            onClick={() => {
              playback.sim?.request(move);
              send({ type: "SHOW", topic: MOVE_TOPICS[move] });
            }}
            title={t("moveTitle", { name: MOVE_NAMES[move], hint: dict().moveHints[move], key: (i + 1) % 10 })}
          >
            {MOVE_NAMES[move]}
          </button>
        ))}
        {playback.canShove !== null && (
          <button
            className="shove"
            disabled={!playing || !playback.canShove}
            onClick={() => playback.sim?.shove()}
            title={t("shoveTitle")}
          >
            {t("shove")}
          </button>
        )}
      </div>

      <div className="moves variations" role="group" aria-label={t("variationsGroup")}>
        <span className="moves-label">{t("variationsLabel")}</span>
        {MANEUVERS.map((maneuver) => (
          <button
            key={maneuver}
            disabled={!ready || !playback.sim?.availableManeuver(maneuver)}
            onClick={() => {
              playback.sim?.requestManeuver(maneuver);
              send({ type: "SHOW", topic: maneuver });
            }}
            title={t("maneuverTitle", { name: MANEUVER_NAMES[maneuver], hint: dict().maneuverHints[maneuver] })}
          >
            {MANEUVER_NAMES[maneuver]}
          </button>
        ))}
        <button
          className="toggle-pill"
          aria-pressed={feintGinga}
          disabled={!ready}
          onClick={() => playback.sim?.feintGinga()}
          title={t("gingaBreakTitle")}
        >
          {t("gingaBreak")}
        </button>
        <button
          className="toggle-pill"
          aria-pressed={feintMode}
          disabled={!playback.sim}
          onClick={() => playback.sim?.toggleFeintMode()}
          title={t("threatModeTitle")}
        >
          {t("threatMode")}
        </button>
      </div>

      <div className="music" role="group" aria-label={t("navigationGroup")}>
        <div className="segmented nav-pad" role="group" aria-label={t("navigateGroup")}>
          <button disabled={!ready} onClick={() => playback.sim?.nudge(0, 0, 1)} title={t("turnLeft")}>
            ↺
          </button>
          <button disabled={!ready} onClick={() => playback.sim?.nudge(1, 0, 0)} title={t("forward")}>
            ↑
          </button>
          <button disabled={!ready} onClick={() => playback.sim?.nudge(-1, 0, 0)} title={t("back")}>
            ↓
          </button>
          <button disabled={!ready} onClick={() => playback.sim?.nudge(0, -1, 0)} title={t("stepLeft")}>
            ←
          </button>
          <button disabled={!ready} onClick={() => playback.sim?.nudge(0, 1, 0)} title={t("stepRight")}>
            →
          </button>
          <button disabled={!ready} onClick={() => playback.sim?.nudge(0, 0, -1)} title={t("turnRight")}>
            ↻
          </button>
        </div>
        <button
          className="music-toggle"
          aria-pressed={orbit}
          disabled={!playback.sim}
          onClick={() => playback.sim?.toggleOrbit()}
          title={t("orbitTitle")}
        >
          {t("orbit")}
        </button>
        <label className="select style-select" title={t("gingaStyleTitle")}>
          <span className="sr-only">{t("gingaStyle")}</span>
          <select
            aria-label={t("gingaStyle")}
            value={gingaStyle}
            onChange={(e) => playback.sim?.setGingaStyle(e.target.value as GingaStyle)}
          >
            {(Object.keys(STYLE_LABELS) as GingaStyle[]).map((style) => (
              <option key={style} value={style}>
                {t(STYLE_LABELS[style])}
              </option>
            ))}
          </select>
        </label>
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
    </div>
  );
}
