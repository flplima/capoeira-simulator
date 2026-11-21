import type * as ToneLib from "tone";
import { renderShake } from "./caxixi";
import { BEATS_PER_BAR, DEFAULT_TOQUE, barEvents, type BarEvent, type Shake, type Stroke, type Toque } from "./toques";
import { renderStroke } from "./wire";

// A synthesized berimbau (no samples) playing a toque in a loop, with its caxixi. Tone.js is
// loaded on first use so the app's initial bundle stays small, and keeps the beat on its
// transport.
//
// Right after Tone.js loads, a few variations of each stroke (wire.ts) and of each caxixi shake
// (caxixi.ts) are rendered into audio buffers. Each event of the bar plays one of them, picked at
// random so repeats don't sound identical, at the exact time the transport gives. A new stroke
// stops what the wire was still ringing, as the stick or the coin touching it would.

type Tone = typeof ToneLib;
type Sound = Stroke | Shake;

/** Variations rendered of each sound. */
const TAKES = 3;
const STROKES: Stroke[] = ["dong", "ding", "chiado"];
const SHAKES: Shake[] = ["strike", "swing"];

interface Voice {
  source: AudioBufferSourceNode;
  gain: GainNode;
  /** When it starts, in the audio context's time. */
  at: number;
}

class Berimbau {
  private tone: Tone | null = null;
  private loading: Promise<Tone> | null = null;
  private context: AudioContext | null = null;
  private sounds: Record<Sound, AudioBuffer[]> | null = null;
  private bus: GainNode | null = null;
  private part: ToneLib.Part | null = null;
  /** The wire's last stroke, which the next one stops. */
  private ringing: Voice | null = null;
  private voices = new Set<Voice>();
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
    // Strokes already handed to the audio clock but not heard yet are taken back.
    const now = this.context?.currentTime ?? 0;
    for (const voice of this.voices) if (voice.at > now) voice.source.stop();
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
    const context = this.context!;
    // Strokes and shakes overlap: at this level their loudest sums stay below full scale.
    const out = new Tone.Gain(0.65).toDestination();
    const room = new Tone.Reverb({ decay: 1.2, wet: 0.14 }).connect(out);
    this.bus = context.createGain();
    Tone.connect(this.bus, room);

    const take = (data: Float32Array) => {
      const buffer = context.createBuffer(1, data.length, context.sampleRate);
      buffer.getChannelData(0).set(data);
      return buffer;
    };
    const renders = [
      ...STROKES.map((stroke) => [stroke, (seed: number) => renderStroke(stroke, context.sampleRate, seed)] as const),
      ...SHAKES.map((shake) => [shake, (seed: number) => renderShake(shake, context.sampleRate, seed)] as const),
    ];
    // One take of each sound now, so the music starts at once; the other takes after, a round at a time.
    const sounds = {} as Record<Sound, AudioBuffer[]>;
    for (const [sound, render] of renders) sounds[sound] = [take(render(1))];
    this.sounds = sounds;
    const more = (seed: number) => {
      for (const [sound, render] of renders) sounds[sound].push(take(render(seed)));
      if (seed < TAKES) setTimeout(() => more(seed + 1));
    };
    setTimeout(() => more(2));
  }

  private play(sound: Sound, at: number, loudness: number): Voice {
    const context = this.context!;
    const takes = this.sounds![sound];
    const source = context.createBufferSource();
    source.buffer = takes[Math.floor(Math.random() * takes.length)];
    const gain = context.createGain();
    // No two strokes of a player are equally loud: within about 1 dB.
    gain.gain.value = loudness * (0.9 + 0.2 * Math.random());
    source.connect(gain).connect(this.bus!);
    const voice = { source, gain, at };
    source.onended = () => {
      gain.disconnect();
      this.voices.delete(voice);
    };
    this.voices.add(voice);
    source.start(at);
    return voice;
  }

  private strike(event: BarEvent, time: number) {
    if (event.stroke) {
      if (this.ringing) {
        this.ringing.gain.gain.setTargetAtTime(0, time, 0.01);
        this.ringing.source.stop(time + 0.1);
      }
      this.ringing = this.play(event.stroke, time, event.strokeGain);
    }
    this.play(event.shake, time, event.shakeGain);
  }

  /** (Re)builds the loop for the current toque, aligned to the transport's bars. */
  private schedule() {
    const T = this.tone;
    if (!T) return;
    const ppq = T.getTransport().PPQ;
    this.part?.dispose();
    this.part = new T.Part<BarEvent & { time: string }>(
      (time, event) => this.strike(event, time),
      barEvents(this.toque).map((event) => ({ ...event, time: `${Math.round(event.beat * ppq)}i` })),
    );
    this.part.loop = true;
    this.part.loopEnd = `${BEATS_PER_BAR * ppq}i`;
    this.part.start(0);
  }
}

export const berimbau = new Berimbau();
