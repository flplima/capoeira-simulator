import { useEffect, useState, useSyncExternalStore } from "react";
import type { EventFromLogic } from "xstate";
import { CapoeiraContext } from "./context";
import type { capoeiraMachine } from "./machine";
import { Playback } from "./playback";
import { Stage } from "./components/Stage";
import { Controls } from "./components/Controls";
import { Info } from "./components/Info";
import { LanguageSwitch } from "./components/LanguageSwitch";
import { t, useLang } from "./i18n/lang";
import { berimbau } from "./music/berimbau";
import { useBerimbau } from "./music/useBerimbau";
import { ask, OFFERED, offeredForKey } from "./offered";

/** Phones and small tablets; keep in step with the compact media query in styles.css. */
const COMPACT_LAYOUT = "(max-width: 900px), (max-height: 520px)";

const KEYS: Record<string, EventFromLogic<typeof capoeiraMachine>> = {
  " ": { type: "TOGGLE_PLAY" },
  ".": { type: "STEP" },
  m: { type: "TOGGLE_MUSIC" },
  "-": { type: "SLOWER" },
  "=": { type: "FASTER" },
  "+": { type: "FASTER" },
};

function useKeyboard(playback: Playback) {
  const actor = CapoeiraContext.useActorRef();
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const playing = actor.getSnapshot().matches("playing");
      // Number keys ask for what the app offers, in the order of the buttons.
      const offered = offeredForKey(e.key);
      if (offered && playback.liveMode === "ginga" && playing) {
        if (playback.sim && ask(playback.sim, offered)) actor.send({ type: "SHOW", topic: offered.topic });
        return;
      }
      const lower = e.key.toLowerCase();
      if (lower === "o") {
        playback.sim?.toggleOrbit();
        return;
      }
      if (lower === "w" || e.key === "ArrowUp") {
        playback.sim?.setNav({ forward: 1 });
        return;
      }
      if (lower === "s" || e.key === "ArrowDown") {
        playback.sim?.setNav({ forward: -1 });
        return;
      }
      if (lower === "a" || e.key === "ArrowLeft") {
        playback.sim?.setNav({ side: -1 });
        return;
      }
      if (lower === "d" || e.key === "ArrowRight") {
        playback.sim?.setNav({ side: 1 });
        return;
      }
      if (lower === "q") {
        playback.sim?.setNav({ turn: 1 });
        return;
      }
      if (lower === "r") {
        playback.sim?.setNav({ turn: -1 });
        return;
      }
      const event = KEYS[e.key.length === 1 ? lower : e.key];
      if (!event) return;
      // Space would otherwise press the focused button too.
      e.preventDefault();
      // Audio may only start inside a user gesture.
      if (event.type === "TOGGLE_MUSIC") berimbau.unlock();
      actor.send(event);
    };
    const onKeyUp = (e: KeyboardEvent) => {
      const lower = e.key.toLowerCase();
      if (lower === "w" || lower === "s" || e.key === "ArrowUp" || e.key === "ArrowDown") {
        playback.sim?.setNav({ forward: 0 });
      } else if (lower === "a" || lower === "d" || e.key === "ArrowLeft" || e.key === "ArrowRight") {
        playback.sim?.setNav({ side: 0 });
      } else if (lower === "q" || lower === "r") {
        playback.sim?.setNav({ turn: 0 });
      }
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [actor, playback]);
}

function Shell() {
  const [playback] = useState(() => new Playback());
  const error = useSyncExternalStore(playback.subscribe, () => playback.error);
  // The info panel starts open beside the stage on a laptop, closed on a phone where it covers the controls.
  const [infoOpen, setInfoOpen] = useState(() => !window.matchMedia(COMPACT_LAYOUT).matches);
  const lang = useLang();
  useKeyboard(playback);
  useBerimbau();
  useEffect(() => {
    document.title = t("title");
  }, [lang]);

  return (
    <div className={`app ${infoOpen ? "info-open" : "info-closed"}`}>
      <div className="stage">
        <Stage playback={playback} />
        <header className="brand">
          <span className="mark" aria-hidden />
          Capoeira <span className="muted">· {t("brandSubtitle")}</span>
        </header>
        <div className="stage-actions">
          <LanguageSwitch />
          <button className="info-toggle" aria-expanded={infoOpen} aria-label={t(infoOpen ? "infoCloseLabel" : "infoOpenLabel")} onClick={() => setInfoOpen(!infoOpen)}>
            {t(infoOpen ? "infoClose" : "infoOpen")}
          </button>
        </div>
        {error && (
          <div className="notice error">
            <p>{t("physicsFailed")}</p>
            <pre>{error}</pre>
          </div>
        )}
      </div>
      <Info />
      <div className="dock">
        <Controls playback={playback} />
        <p className="keys">
          <kbd>{t("keySpace")}</kbd> {t("keyPlay")} · <kbd>1</kbd>–<kbd>{OFFERED.length}</kbd> {t("keyMoves")} · <kbd>WASD</kbd>/<kbd>Q</kbd><kbd>R</kbd> {t("keyWalk")} ·{" "}
          <kbd>O</kbd> {t("keyOrbit")} · <kbd>M</kbd> {t("keyBerimbau")}
        </p>
      </div>
    </div>
  );
}

export function App() {
  return (
    <CapoeiraContext.Provider>
      <Shell />
    </CapoeiraContext.Provider>
  );
}
