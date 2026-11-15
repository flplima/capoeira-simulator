import type * as ToneLib from "tone";
import { BEATS_PER_BAR, DEFAULT_TOQUE, type Stroke, type Toque } from "./toques";

// A synthesized berimbau (no samples) playing a toque in a loop, with a soft
// caxixi on the eighth notes. Tone.js is loaded on first use so the app's
// initial bundle stays small.
//
// The voice: a plucked steel wire (Karplus–Strong) for the two notes, run
// through a band-pass "gourd" whose centre sweeps up after each strike — the
// wah of the cabaça leaving the belly. The chiado is a short, bright buzz: the
// coin resting on the wire, so mostly noise with a metallic rattle.

type Tone = typeof ToneLib;

/** Wire pitch (Hz) of the open note; the closed note is a whole tone above. */
const DONG = 110;
const DING = DONG * Math.pow(2, 2 / 12);

class Berimbau {
  private tone: Tone | null = null;
  private loading: Promise<Tone> | null = null;
  private context: AudioContext | null = null;
  private voice: { pluck: ToneLib.PluckSynth; gourd: ToneLib.Filter; buzz: ToneLib.NoiseSynth; rattle: ToneLib.MetalSynth; caxixi: ToneLib.NoiseSynth } | null = null;
  private part: ToneLib.Part | null = null;
  private shaker: ToneLib.Loop | null = null;
  private toque: Toque = DEFAULT_TOQUE;
  private bpm = DEFAULT_TOQUE.bpm;
  private wanted = false;

  /**
   * Browsers only let audio start inside a user gesture, and loading Tone.js
   * takes longer than a gesture lasts. Call this synchronously from the click
   * or key handler: it creates and resumes the audio context right away.
   */
  unlock() {
    if (!this.context) this.context = new AudioContext({ latencyHint: "interactive" });
    if (this.context.state !== "running") void this.context.resume();
  }

  get playing() {
    return this.wanted && this.tone?.getTransport().state === "started";
  }

  async start() {
    this.wanted = true;
    const Tone = await this.load();
    if (!this.wanted) return;
    await Tone.start();
    const transport = Tone.getTransport();
    if (transport.state === "started") return;
    transport.bpm.value = this.bpm;
    this.schedule();
    // Start at the top of the bar, so the first thing heard is the chiado on beat one.
    transport.start("+0.05", 0);
  }

  stop() {
    this.wanted = false;
    this.tone?.getTransport().stop();
  }

  setToque(toque: Toque) {
    if (toque.id === this.toque.id) return;
    this.toque = toque;
    if (this.tone) this.schedule();
  }

  /** Changes the tempo smoothly, without restarting the bar. */
  setTempo(bpm: number) {
    this.bpm = bpm;
    this.tone?.getTransport().bpm.rampTo(bpm, 0.25);
  }

  /** Where in the bar the music is (0 = beat one, as heard), in [0, 1); null when it isn't playing. */
  barPhase(): number | null {
    if (!this.tone || !this.playing) return null;
    const transport = this.tone.getTransport();
    const raw = this.tone.getContext().rawContext as AudioContext;
    // What is audible now, not what the scheduler is preparing ahead.
    const heard = Math.max(0, raw.currentTime - (raw.outputLatency || raw.baseLatency || 0));
    const beats = transport.getTicksAtTime(heard) / transport.PPQ;
    return (((beats % BEATS_PER_BAR) + BEATS_PER_BAR) % BEATS_PER_BAR) / BEATS_PER_BAR;
  }

  private load(): Promise<Tone> {
    this.loading ??= import("tone").then((Tone) => {
      this.unlock();
      Tone.setContext(this.context!);
      this.build(Tone);
      this.tone = Tone;
      return Tone;
    });
    return this.loading;
  }

  private build(Tone: Tone) {
    const out = new Tone.Gain(0.8).toDestination();
    const room = new Tone.Reverb({ decay: 1.2, wet: 0.14 }).connect(out);
    const body = new Tone.Filter({ type: "peaking", frequency: 240, Q: 1.2, gain: 8 }).connect(room);
    const gourd = new Tone.Filter({ type: "bandpass", frequency: 700, Q: 2.2 }).connect(body);
    const pluck = new Tone.PluckSynth({ attackNoise: 2.2, dampening: 5200, resonance: 0.985, volume: 4 });
    pluck.connect(gourd);
    // A little of the wire straight out, so it keeps its metallic edge.
    pluck.connect(new Tone.Gain(0.25).connect(room));

    const buzzFilter = new Tone.Filter({ type: "bandpass", frequency: 3400, Q: 0.9 }).connect(room);
    const buzz = new Tone.NoiseSynth({ noise: { type: "white" }, envelope: { attack: 0.001, decay: 0.045, sustain: 0, release: 0.02 }, volume: -10 });
    buzz.connect(buzzFilter);
    const rattle = new Tone.MetalSynth({
      envelope: { attack: 0.001, decay: 0.07, release: 0.02 },
      harmonicity: 3.1,
      modulationIndex: 18,
      resonance: 2600,
      octaves: 1.2,
      volume: -24,
    });
    rattle.frequency.value = DONG * 2;
    rattle.connect(room);

    const shakerFilter = new Tone.Filter({ type: "highpass", frequency: 5200 }).connect(out);
    const caxixi = new Tone.NoiseSynth({ noise: { type: "pink" }, envelope: { attack: 0.006, decay: 0.07, sustain: 0, release: 0.03 }, volume: -20 });
    caxixi.connect(shakerFilter);

    this.voice = { pluck, gourd, buzz, rattle, caxixi };
  }

  private strike(stroke: Stroke, time: number) {
    const v = this.voice!;
    if (stroke === "chiado") {
      v.buzz.triggerAttackRelease(0.05, time);
      v.rattle.triggerAttackRelease(0.05, time, 0.6);
      return;
    }
    // The gourd opens after the strike: its band sweeps up, then settles.
    const f = v.gourd.frequency;
    f.cancelScheduledValues(time);
    f.setValueAtTime(stroke === "dong" ? 520 : 650, time);
    f.exponentialRampToValueAtTime(stroke === "dong" ? 1500 : 1300, time + 0.16);
    f.exponentialRampToValueAtTime(900, time + 0.5);
    v.pluck.triggerAttack(stroke === "dong" ? DONG : DING, time);
  }

  /** (Re)builds the loop for the current toque, aligned to the transport's bars. */
  private schedule() {
    const T = this.tone;
    if (!T) return;
    const ppq = T.getTransport().PPQ;
    this.part?.dispose();
    this.part = new T.Part<{ time: string; stroke: Stroke }>(
      (time, event) => this.strike(event.stroke, time),
      this.toque.pattern.map((p) => ({ time: `${Math.round(p.beat * ppq)}i`, stroke: p.stroke })),
    );
    this.part.loop = true;
    this.part.loopEnd = `${BEATS_PER_BAR * ppq}i`;
    this.part.start(0);
    if (!this.shaker) {
      this.shaker = new T.Loop((time) => {
        // Accent on the beat, lighter off the beat.
        const eighth = Math.round(T.getTransport().getTicksAtTime(time) / (ppq / 2));
        this.voice!.caxixi.triggerAttackRelease(0.06, time, eighth % 2 === 0 ? 1 : 0.55);
      }, "8n").start(0);
    }
  }
}

export const berimbau = new Berimbau();
