/** The English interface. Capoeira names (moves, maneuvers, toques, berimbau, roda, ginga) stay Portuguese. */
export const en = {
  title: "Capoeira · real physics",
  brandSubtitle: "real physics, trained by machine learning",
  languageGroup: "Language",
  infoClose: "Close",
  infoOpen: "About the move",
  infoCloseLabel: "Close the info panel",
  infoOpenLabel: "Open the info panel",
  physicsFailed: "Real-time physics did not load.",
  keySpace: "space",
  keyPlay: "play",
  keyMoves: "moves",
  keyWalk: "walk/turn",
  keyOrbit: "orbit",
  keyBerimbau: "berimbau",

  play: "Play",
  pause: "Pause",
  playPauseTitle: "Play / pause (space)",
  step: "One step",
  stepTitle: "One control step, {dt} s (.)",
  speed: "Speed",
  playerGroup: "Playback and berimbau",
  movesGroup: "Moves",
  moveTitle: "{name}: {hint}; the leg and the moment are chosen by the training ({key})",
  maneuverTitle: "{name}: {hint} ({key})",
  berimbauToggle: "♪ Berimbau",
  berimbauTitle: "Play the berimbau (M)",
  toque: "Toque",
  tempoGroup: "Tempo",
  slower: "Slower",
  faster: "Faster",
  tempoTitle: "Tempo (beats per minute)",

  trainedMovements: "Trained movements",
  also: "also",
  howTo: "How to do it",
  tips: "Tips",
  sources: "Sources",


  moveHints: {
    "meia-lua-de-frente": "outside-in arc with the leg extended",
    joelhada: "short, direct knee strike",
    esquiva: "drops over the front leg, the back leg slides out to the side",
    cocorinha: "compact squat protecting the face",
  },
  maneuverHints: {
    "giro-de-base": "360° turn standing on the support foot",
  },
  toqueDescriptions: {
    angola: "Slow and cadenced: a low, wide, flowing ginga close to the floor.",
    "sao-bento-pequeno": "Angola inverted, faster: inside game, medium ginga.",
    "sao-bento-grande": "Fast and upright: a high, short, rhythmic ginga, ready for the kick.",
  },

  sheetTitle: "{id} · sheet",
  sheetViews: { front: "front", side: "side", back: "back", top: "top", "3/4": "3/4" },
};

export type Dictionary = typeof en;
