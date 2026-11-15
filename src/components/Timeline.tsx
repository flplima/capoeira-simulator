import { useEffect, useRef } from "react";
import { decimal, t, useLang } from "../i18n/lang";
import type { Playback } from "../playback";

/** The ginga cycle, a bar of four beats, with a playhead and the status line, updated without re-rendering React. */
export function Timeline({ playback }: { playback: Playback }) {
  const head = useRef<HTMLDivElement>(null);
  const label = useRef<HTMLSpanElement>(null);
  const clock = useRef<HTMLSpanElement>(null);
  const ticks = useRef<HTMLDivElement>(null);
  useLang();

  useEffect(() => {
    let frame = 0;
    let lastBeat = -1;
    const tick = () => {
      const s = playback.status;
      head.current!.style.transform = `scaleX(${s.time / s.duration})`;
      clock.current!.textContent = playback.sim ? `${decimal(s.time, 2)} / ${decimal(s.duration, 2)} s` : "";
      if (label.current!.textContent !== s.label) label.current!.textContent = s.label;
      const beat = playback.sim ? Math.floor((4 * s.time) / s.duration) : -1;
      if (beat !== lastBeat) {
        ticks.current?.querySelectorAll("span").forEach((b, i) => b.classList.toggle("current", i === beat));
        lastBeat = beat;
      }
      frame = requestAnimationFrame(tick);
    };
    tick();
    return () => cancelAnimationFrame(frame);
  }, [playback]);

  return (
    <div className="timeline">
      <div className="timeline-meta">
        <span ref={label} className="timeline-label" />
        <span ref={clock} className="timeline-clock" />
      </div>
      <div className="track">
        <div ref={head} className="progress" />
        <div ref={ticks}>
          {[0, 1, 2, 3].map((beat) => (
            <span key={beat} className="tick" style={{ left: `${beat * 25}%` }} title={t("beat", { n: beat + 1 })} />
          ))}
        </div>
      </div>
    </div>
  );
}
