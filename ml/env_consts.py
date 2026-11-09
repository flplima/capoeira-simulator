"""Constants of the capoeira environment: control, tempo, the moves and their blending, falling and getting up."""

FRAME_SKIP = 15  # 0.002 s × 15 = 0.03 s per action
ACTION_SCALE = 0.5  # rad
EFFECTORS = ["foot_l", "foot_r", "hand_l", "hand_r", "head"]
ALLOWED_CONTACT = {"foot_l", "foot_r", "hand_l", "hand_r"}
# While falling and getting up, anything but the head and neck may touch the floor.
NEVER_CONTACT = {"head", "neck"}

MIN_BPM, MAX_BPM = 60.0, 130.0
TEMPO_SKEW = 1.3  # >1 favours slow tempos when sampling (see GingaEnv._random_tempo): slow is the hardest to keep, fast still needs practice
TEMPO_CHANGE_EVERY = 8.0  # s, on average, while training: the toque can change mid-game


def tempo_input(bpm: float) -> float:
    """The tempo as the policy sees it, about −1 (slowest) to 1 (fastest)."""
    return (bpm - 95.0) / 35.0


KICK_MOVES = ["martelo", "bencao", "meia-lua-de-frente", "queixada", "ponteira", "joelhada", "meia-lua-de-compasso"]
# While a kick is under way, this share of the reward pays for the kicking foot being where the reference's is,
# relative to the hips but not to their heading (GingaEnv._reward "strike")…
STRIKE_WEIGHT = 0.15
# …and this share for the support foot turning as the reference's does ("pivot"): pointing the same way, its heel
# as high (a martelo turns on the ball, heel 7 cm up) and its ball where the reference's is. Without the heel and
# the ball the body turned its foot flat and slid it 0.3–0.7 m a kick.
PIVOT_WEIGHT = 0.2
# Every tracking term scores exp(-error²/σ), and σ is a tolerance the training sets itself: the running mean of
# that term's error, only ever tightened (KungfuBot, Xie et al. 2025). At first everything is loose; as the body
# gets closer the tolerance follows it down, and an error it never closes — a reference that is itself an
# estimate, from one camera or keyframes — stays loose instead of costing reward forever. One set of tolerances
# for the ginga (and the floor) and one for each move on each leg (SIGMA_ROWS): tolerances shared between moves
# let the one that learns first set a bar the others score zero against, and never learn (the right martelo at
# 85 %, the left frozen at 10 %; the martelo frozen at 0 among keyframe kicks at 90 %).
# The errors are summed squares: joint angles with the root's turn (rad²), joint velocities, the hands, feet and
# head as the body faces (m², per effector), the root (m²), the feet on the triangle (m²), the kicking foot from
# the hips (m²), and the support foot's heading (rad²), heel height (m²) and ball position (m²).
TERMS = ["pose", "vel", "ee", "root", "feet", "strike", "turn", "heel", "ball"]
KICK_TERMS = TERMS.index("strike")  # from here on, a kick's own terms: counted only inside a kick
SIGMA_INIT = 1e3  # loose enough that the first iteration's errors replace it
SIGMA_EMA = 0.2  # share of an iteration's mean error that moves the running mean
SIGMA_FLOOR = 1e-6  # a tolerance never tightens below this (squared units)
# Iterations after a (re)start during which a tolerance follows its running mean both ways before it only tightens:
# the first iterations' errors all come from fresh starts, with the body still on the reference, and run low.
SIGMA_WARMUP = 5
DODGES = ["esquiva", "esquiva-baixa", "cocorinha"]  # see esquiva.py
MOVES = KICK_MOVES + DODGES
SIGMA_ROWS = 1 + 2 * len(MOVES)  # tolerance rows: the ginga, then each move on the left leg and on the right
EXTRA_MOVES = ["meia-lua-de-frente", "queixada", "ponteira", "joelhada", "meia-lua-de-compasso", "esquiva-baixa", "cocorinha"]
# What each practice skill asks for (GingaEnv skill), and which network plays each move (brain.py).
SKILL_MOVES = {"kicks": KICK_MOVES, "esquiva": DODGES, "all": MOVES}
# A kick (or an esquiva) blends in from the ginga over BLEND_IN seconds and back out over BLEND_OUT.
BLEND_IN, BLEND_OUT = 0.35, 0.35
# Where in the ginga cycle each leg can start a kick: around the base with that leg back.
ENTRY_CENTRE = {"r": 0.0, "l": 0.5}
ENTRY_WINDOW = 0.1
KICK_EVERY = 4.0  # s, on average, while training
ANCHOR_LIMIT = 0.25  # m: furthest a kick follows its support foot away from the corner
# s of ginga after a kick or an esquiva in a practice episode. The move's network drives the ginga that long
# before handing back (brain.MOVE_HANDBACK), so it practises getting all the way back into a settled ginga —
# the ginga network fell from what it was handed after 1 s.
MOVE_EPISODE_TAIL = 3.0
# s of ginga after a kick before it counts as done standing (curriculum, best checkpoint, check.py): bodies that
# ended a turning kick twisted fell 0.3–1.2 s into the ginga, after the kick had already counted.
KICK_SETTLE = 1.0
GINGA_FIRST = 0.3  # share of kick and esquiva practice episodes that start in the plain ginga and wait for the entry

LANDINGS = ["quatro", "negativa", "joelhos"]  # see floor.py
# Why an episode ended (logged by train.py as shares of the episodes that ended).
ENDS = ["time", "contact", "low", "drift", "stuck", "dive"]
FALL_TIME = 0.5  # s from losing balance to the landing pose (see _recover_target)
TROUBLE_TILT = 0.65  # rad: the hips tipped this far from the reference's (heading aside: a martelo's pivot is no fall)…
TROUBLE_DROP = 0.25  # m: …or this far below its height, and the body is falling
# Outside a recovery the hips below LOW_HIPS or the head below LOW_HEAD (m) is a fall — or, where the reference
# goes lower than that (an esquiva), LOW_BELOW further down than the reference's.
LOW_HIPS, LOW_HEAD, LOW_BELOW = 0.4, 0.8, 0.3
RECOVER_TAIL = 2.0  # s of ginga after getting up in a recovery-practice episode
# While falling and getting up the arms brace — a person catching their weight on their hands tenses them:
# the arm motors' stiffness (N·m/rad) and torque limit (N·m) become these (the ginga's are ~half; damping
# stays critical). The ginga and the kicks keep the relaxed arms they were trained with.
STRONG_ARMS = {"shoulder": (300.0, 100.0), "elbow": (250.0, 80.0)}
# On the way up, the reference waits for a body that lags behind it: its clock slows from full speed with the
# hips WAIT_FROM (m) off the reference's to MIN_PACE of it at WAIT_FROM + WAIT_SPAN — getting up from the floor
# is done at the body's own pace, not the clip's.
WAIT_FROM, WAIT_SPAN, MIN_PACE = 0.15, 0.3, 0.2
LOST = 0.6  # m: on the floor, this far from the way up and the body is lost (only after landing)
STUCK = 1.0  # s: after landing, the way up waiting at its slowest this long and the body is stuck on the floor
# Fall curriculum (skill "recover"): a recorded fall starts with its velocity scaled to one of these; each
# environment starts with the gentlest and moves up once most of its falls at that speed end up standing.
FALL_SPEEDS = (0.0, 0.35, 0.7, 1.0)
BENT = 0.9  # rad: a trunk already bent forward this far (hips to head) lands on hands and knees whichever way it goes
GRAVITY = 9.81  # m/s²
# Going down and just after landing, the head this far below the reference's ends the episode (skill recover): a
# body that keeps its hips up and pitches over them dives head first, and only hits the floor a moment later.
DIVE, DIVE_UNTIL = 0.3, 0.5  # m; s after landing
# Going down, no pose has these below the floor (a knee: its shin's surface, this far below the joint's site).
GROUND_SITES = ["heel_l", "toe_l", "heel_r", "toe_r", "knee_l", "knee_r", "fingers_l", "fingers_r"]
KNEE_CLEAR = 0.054
HEAD_ABOVE = 0.1  # m: going down, the head stays at least this far above the hips (every landing has it ~0.2 m up or more)
