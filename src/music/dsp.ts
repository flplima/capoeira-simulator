// Small signal helpers shared by the berimbau's offline renderers.

/** A seeded random generator in [0, 1) (mulberry32), so a render can be repeated exactly. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Runs `input` through a band-pass biquad (0 dB at its centre) and adds the result, times `gain`, to `out`. */
export function addBandpass(input: Float32Array, out: Float32Array, hz: number, q: number, gain: number, sampleRate: number) {
  const w = (2 * Math.PI * hz) / sampleRate;
  const alpha = Math.sin(w) / (2 * q);
  const a0 = 1 + alpha;
  const b0 = alpha / a0;
  const a1 = (-2 * Math.cos(w)) / a0;
  const a2 = (1 - alpha) / a0;
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < input.length; i++) {
    const x = input[i];
    const y = b0 * (x - x2) - a1 * y1 - a2 * y2;
    x2 = x1;
    x1 = x;
    y2 = y1;
    y1 = y;
    out[i] += gain * y;
  }
}

/** Adds a burst of noise decaying with time constant `tau` (s), starting at sample `at`, to `out`. */
export function addGrain(out: Float32Array, at: number, amp: number, tau: number, sampleRate: number, random: () => number) {
  const decay = Math.exp(-1 / (tau * sampleRate));
  const end = Math.min(out.length, at + Math.ceil(tau * sampleRate * 7));
  let a = amp;
  for (let i = at; i < end; i++, a *= decay) out[i] += a * (2 * random() - 1);
}

/** Fades the last `seconds` of a buffer to silence and scales it so its loudest sample is `peak`. */
export function finish(out: Float32Array, seconds: number, peak: number, sampleRate: number): Float32Array {
  const fade = Math.min(out.length, Math.round(seconds * sampleRate));
  for (let i = 0; i < fade; i++) out[out.length - 1 - i] *= i / fade;
  let max = 0;
  for (let i = 0; i < out.length; i++) max = Math.max(max, Math.abs(out[i]));
  const scale = max > 0 ? peak / max : 0;
  for (let i = 0; i < out.length; i++) out[i] *= scale;
  return out;
}
