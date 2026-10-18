import { useEffect, useState, useSyncExternalStore } from "react";
import type { EventFromLogic } from "xstate";
import { CapoeiraContext } from "./context";
import type { capoeiraMachine } from "./machine";
import { Playback } from "./playback";
import { Stage } from "./components/Stage";
import { Controls, MOVE_TOPICS } from "./components/Controls";
import { Timeline } from "./components/Timeline";
import { Info } from "./components/Info";
import { berimbau } from "./music/berimbau";
import { useBerimbau } from "./music/useBerimbau";
import type { Move } from "./sim/simulation";

const KEYS: Record<string, EventFromLogic<typeof capoeiraMachine>> = {
  " ": { type: "TOGGLE_PLAY" },
  ".": { type: "STEP" },
  m: { type: "TOGGLE_MUSIC" },
  "-": { type: "SLOWER" },
  "=": { type: "FASTER" },
  "+": { type: "FASTER" },
};

/** Number keys ask for the moves, in the order of the move buttons. */
const MOVE_KEYS: Record<string, Move> = {
  "1": "martelo",
  "2": "bencao",
  "3": "meia-lua-de-frente",
  "4": "queixada",
  "5": "ponteira",
  "6": "joelhada",
  "7": "meia-lua-de-compasso",
  "8": "esquiva",
  "9": "esquiva-baixa",
  "0": "cocorinha",
};

function useKeyboard(playback: Playback) {
  const actor = CapoeiraContext.useActorRef();
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const playing = actor.getSnapshot().matches("playing");
      const move = MOVE_KEYS[e.key];
      if (move && playback.liveMode === "ginga" && playing) {
        if (playback.sim?.request(move)) actor.send({ type: "SHOW", topic: MOVE_TOPICS[move] });
        return;
      }
      const lower = e.key.toLowerCase();
      if (lower === "e" && playback.canShove && playing) {
        playback.sim?.shove();
        return;
      }
      if (lower === "f" && playback.liveMode === "ginga" && playing) {
        playback.sim?.feintGinga();
        return;
      }
      if (lower === "g") {
        playback.sim?.toggleFeintMode();
        return;
      }
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
  useKeyboard(playback);
  useBerimbau();

  return (
    <div className="app">
      <Stage playback={playback} />
      <header className="brand">
        <span className="mark" aria-hidden />
        Capoeira <span className="muted">· física real, treinada por aprendizado de máquina</span>
      </header>
      {error && (
        <div className="notice error">
          <p>A física em tempo real não carregou.</p>
          <pre>{error}</pre>
        </div>
      )}
      <Info />
      <div className="dock">
        <Timeline playback={playback} />
        <Controls playback={playback} />
        <p className="keys">
          <kbd>espaço</kbd> tocar · <kbd>1</kbd>–<kbd>0</kbd> golpes · <kbd>WASD</kbd>/<kbd>Q</kbd><kbd>R</kbd> andar/girar ·{" "}
          <kbd>F</kbd> quebra · <kbd>G</kbd> ameaça · <kbd>O</kbd> orbitar · <kbd>E</kbd> empurrão · <kbd>M</kbd> berimbau
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
