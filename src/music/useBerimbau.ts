import { useEffect } from "react";
import { CapoeiraContext } from "../context";
import { berimbau } from "./berimbau";
import { TOQUES } from "./toques";

/** Keeps the berimbau in step with the app: sounding while music is on and playing, at the toque's tempo times the playback speed. */
export function useBerimbau() {
  const music = CapoeiraContext.useSelector((s) => s.context.music);
  const playing = CapoeiraContext.useSelector((s) => s.matches("playing"));
  const toque = CapoeiraContext.useSelector((s) => s.context.toque);
  const bpm = CapoeiraContext.useSelector((s) => s.context.bpm);
  const speed = CapoeiraContext.useSelector((s) => s.context.speed);

  useEffect(() => {
    const t = TOQUES.find((x) => x.id === toque);
    if (t) berimbau.setToque(t);
  }, [toque]);
  // Slow motion slows the music too, so the ginga stays on the beat.
  useEffect(() => berimbau.setTempo(bpm * speed), [bpm, speed]);
  useEffect(() => {
    if (music && playing) void berimbau.start();
    else berimbau.stop();
  }, [music, playing]);
  useEffect(() => () => berimbau.stop(), []);
}
