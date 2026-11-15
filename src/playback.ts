import { t } from "./i18n/lang";
import type { LiveGinga } from "./sim/simulation";

/** What the live simulation is doing, for the UI: loading, failed to load, gingando, waiting to kick, kicking, down. */
export type LiveMode = "loading" | "failed" | "ginga" | "waiting" | "kick" | "down";

/** The live simulation the stage runs, once its engine and networks have loaded. */
export class Playback {
  private live: LiveGinga | null = null;
  private failure: string | null = null;
  private listeners = new Set<() => void>();
  private unsubscribe: (() => void) | null = null;

  get sim() {
    return this.live;
  }

  set sim(live: LiveGinga | null) {
    this.unsubscribe?.();
    this.live = live;
    this.unsubscribe = live ? live.subscribe(() => this.emit()) : null;
    this.emit();
  }

  /** Why the simulation didn't load, if it didn't. */
  get error() {
    return this.failure;
  }

  set error(message: string | null) {
    this.failure = message;
    this.emit();
  }

  get liveMode(): LiveMode {
    return this.live ? this.live.mode : this.failure ? "failed" : "loading";
  }

  /** For useSyncExternalStore: called when the simulation starts, stops, or changes mode. */
  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    return () => void this.listeners.delete(listener);
  };

  private emit() {
    for (const listener of this.listeners) listener();
  }

  get status() {
    if (this.live) return this.live.status;
    return { time: 0, duration: 1, label: t(this.failure ? "physicsUnavailable" : "physicsLoading") };
  }
}
