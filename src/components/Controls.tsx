import { useSyncExternalStore } from "react";
import { CapoeiraContext } from "../context";
import type { Playback } from "../playback";
import {
  MANEUVER_NAMES,
  MANEUVERS,
  MOVE_NAMES,
  MOVES,
  type GingaStyle,
  type Maneuver,
  type Move,
} from "../sim/types";
import { SPEEDS, type Topic } from "../machine";
import { berimbau } from "../music/berimbau";
import { MAX_BPM, MIN_BPM, TOQUES } from "../music/toques";

const BPM_STEP = 2;

const MOVE_HINTS: Record<Move, string> = {
  martelo: "chute circular com o peito do pé",
  bencao: "empurrão frontal com a sola",
  "meia-lua-de-frente": "arco de fora para dentro com a perna estendida",
  queixada: "arco de dentro para fora",
  ponteira: "chute frontal rápido com a bola do pé",
  joelhada: "golpe curto e direto com o joelho",
  "meia-lua-de-compasso": "giro por trás apoiando as mãos no chão",
  esquiva: "desce sobre a perna da frente, a de trás desliza para o lado",
  "esquiva-baixa": "desce rente ao chão sobre a perna da frente",
  cocorinha: "agachamento compacto protegendo o rosto",
};

const MANEUVER_HINTS: Record<Maneuver, string> = {
  "troca-de-base": "passo rápido na paralela invertendo a perna de trás",
  "giro-de-base": "giro de 360° em pé sobre o pé de apoio",
  role: "giro baixo de 270° perto do chão deslocando pela roda",
  "passagem-de-negativa": "descida baixa com giro de 90° e saída na base oposta",
};

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
  loading: "carregando…",
  failed: "indisponível",
  waiting: "esperando o momento…",
  kick: "em movimento…",
  down: "no chão…",
  ginga: "golpes",
} as const;

export function Controls({ playback }: { playback: Playback }) {
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
  const toque = TOQUES.find((t) => t.id === toqueId) ?? TOQUES[0];
  const send = actor.send;

  return (
    <div className="controls">
      <div className="transport">
        <button className="play" onClick={() => send({ type: "TOGGLE_PLAY" })} aria-label={playing ? "Pausar" : "Tocar"} title="Tocar / pausar (espaço)">
          {playing ? "❚❚" : "▶"}
        </button>
        <button onClick={() => send({ type: "STEP" })} disabled={!playback.sim} aria-label="Um passo" title="Um passo de controle, 0,03 s (.)">
          ⏭
        </button>
      </div>

      <div className="segmented" role="group" aria-label="Velocidade">
        {SPEEDS.map((s) => (
          <button key={s} aria-pressed={s === speed} onClick={() => send({ type: "SET_SPEED", speed: s })}>
            {s}×
          </button>
        ))}
      </div>

      <div className="moves" role="group" aria-label="Golpes">
        <span className="moves-label">{MODE_LABELS[live]}</span>
        {MOVES.map((move, i) => (
          <button
            key={move}
            disabled={!ready || !playback.sim?.available(move)}
            onClick={() => {
              playback.sim?.request(move);
              send({ type: "SHOW", topic: MOVE_TOPICS[move] });
            }}
            title={`${MOVE_NAMES[move]} — ${MOVE_HINTS[move]}; a perna e o momento são escolhidos pelo treino (${(i + 1) % 10})`}
          >
            {MOVE_NAMES[move]}
          </button>
        ))}
        {playback.canShove !== null && (
          <button
            className="shove"
            disabled={!playing || !playback.canShove}
            onClick={() => playback.sim?.shove()}
            title="Empurrão — um empurrão de um lado qualquer: o corpo cai como um capoeirista — nas mãos e joelhos, na queda de quatro ou na negativa —, levanta e volta para a ginga (E)"
          >
            Empurrão
          </button>
        )}
      </div>

      <div className="moves variations" role="group" aria-label="Variações de ginga e deslocamentos">
        <span className="moves-label">jogo & roda</span>
        {MANEUVERS.map((maneuver) => (
          <button
            key={maneuver}
            disabled={!ready || !playback.sim?.availableManeuver(maneuver)}
            onClick={() => {
              playback.sim?.requestManeuver(maneuver);
              send({ type: "SHOW", topic: maneuver });
            }}
            title={`${MANEUVER_NAMES[maneuver]} — ${MANEUVER_HINTS[maneuver]}`}
          >
            {MANEUVER_NAMES[maneuver]}
          </button>
        ))}
        <button
          className="toggle-pill"
          aria-pressed={feintGinga}
          disabled={!ready}
          onClick={() => playback.sim?.feintGinga()}
          title="Quebra de ginga — finta a passagem na paralela e retorna à mesma base (F)"
        >
          Quebra de ginga
        </button>
        <button
          className="toggle-pill"
          aria-pressed={feintMode}
          disabled={!playback.sim}
          onClick={() => playback.sim?.toggleFeintMode()}
          title="Modo ameaça — qualquer golpe pedido sai como finta (45% de extensão) e recolhe para a ginga (G)"
        >
          Modo ameaça
        </button>
      </div>

      <div className="music" role="group" aria-label="Navegação e Berimbau">
        <div className="segmented nav-pad" role="group" aria-label="Navegar pela roda">
          <button disabled={!ready} onClick={() => playback.sim?.nudge(0, 0, 1)} title="Girar à esquerda (Q)">
            ↺
          </button>
          <button disabled={!ready} onClick={() => playback.sim?.nudge(1, 0, 0)} title="Avançar (W / ↑)">
            ↑
          </button>
          <button disabled={!ready} onClick={() => playback.sim?.nudge(-1, 0, 0)} title="Recuar (S / ↓)">
            ↓
          </button>
          <button disabled={!ready} onClick={() => playback.sim?.nudge(0, -1, 0)} title="Deslocar à esquerda (A / ←)">
            ←
          </button>
          <button disabled={!ready} onClick={() => playback.sim?.nudge(0, 1, 0)} title="Deslocar à direita (D / →)">
            →
          </button>
          <button disabled={!ready} onClick={() => playback.sim?.nudge(0, 0, -1)} title="Girar à direita (R)">
            ↻
          </button>
        </div>
        <button
          className="music-toggle"
          aria-pressed={orbit}
          disabled={!playback.sim}
          onClick={() => playback.sim?.toggleOrbit()}
          title="Circular pela roda mantendo a frente para o centro (O)"
        >
          ◎ Orbitar roda
        </button>
        <label className="select style-select" title="Estilo postural da ginga (Angola baixa, Padrão ou Regional alta)">
          <span className="sr-only">Estilo da ginga</span>
          <select
            aria-label="Estilo da ginga"
            value={gingaStyle}
            onChange={(e) => playback.sim?.setGingaStyle(e.target.value as GingaStyle)}
          >
            <option value="auto">Estilo: Auto (Toque)</option>
            <option value="angola">Estilo: Angola (Baixa)</option>
            <option value="padrao">Estilo: Padrão</option>
            <option value="regional">Estilo: Regional (Alta)</option>
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
          title="Tocar o berimbau (M)"
        >
          ♪ Berimbau
        </button>
        <label className="select toque" title={toque.description}>
          <span className="sr-only">Toque</span>
          <select value={toque.id} onChange={(e) => send({ type: "SET_TOQUE", id: e.target.value })}>
            {TOQUES.map((t) => (
              <option key={t.id} value={t.id} title={t.description}>
                {t.name} · {t.bpm} bpm
              </option>
            ))}
          </select>
        </label>
        <div className="segmented tempo" role="group" aria-label="Andamento">
          <button onClick={() => send({ type: "SET_BPM", bpm: bpm - BPM_STEP })} disabled={bpm <= MIN_BPM} aria-label="Mais lento" title="Mais lento">
            −
          </button>
          <output aria-live="polite" title="Andamento (batidas por minuto)">
            {bpm} bpm
          </output>
          <button onClick={() => send({ type: "SET_BPM", bpm: bpm + BPM_STEP })} disabled={bpm >= MAX_BPM} aria-label="Mais rápido" title="Mais rápido">
            +
          </button>
        </div>
      </div>
    </div>
  );
}
