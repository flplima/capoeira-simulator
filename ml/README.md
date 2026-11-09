# Ginga com física real e aprendizado de máquina

Uma segunda versão da ginga, em que o corpo não segue quadros-chave: um
boneco sólido simulado com massa, articulações, torque limitado, atrito e
gravidade (MuJoCo) é controlado por uma rede neural treinada por reforço para
gingar como um mestre num vídeo de referência — sem cair, pisando no
triângulo e no andamento do berimbau. Nada é gravado: o navegador roda o
próprio MuJoCo (WebAssembly) e a rede neural, e o corpo decide cada movimento
em tempo real (chave **Física real (ML)**). Do meio da ginga, o corpo também
solta dois golpes quando pedido — **martelo** e **bênção** — e volta à ginga;
a perna e o momento de entrar no golpe também vêm do treino.

## Pipeline

```
vídeo de referência (duas câmeras: frente e lado)
   │  capture.py    MediaPipe Pose (rede neural) → 33 pontos 3D por quadro
   ▼
pontos 3D
   │  retarget.py   ajuste do esqueleto do boneco aos pontos (mínimos quadrados
   │                sobre as direções dos segmentos) → ângulos por quadro
   ▼
ângulos de cada câmera
   │  reference.py  fusão das câmeras (frente para o plano frontal, lado para o
   │                sagital), alinhamento de fase, simetria esquerda/direita,
   │                filtro de Fourier, pé de apoio carregando o corpo; depois,
   │                para cada andamento, o estilo (altura, inclinação, passo),
   │                IK das pernas para os pés pisarem no triângulo e IK dos
   │                braços para a guarda não atravessar a cabeça
   ▼
ciclos de referência (60, 78, 96, 114 e 130 bpm; um ciclo = um compasso de 4 tempos)
   +
golpes (kicks.py): martelo e bênção dos quadros-chave do app (ajustados às
   duas câmeras do vídeo), cada perna, plantados no triângulo por IK
   +
chão (floor.py): queda de quatro e negativa — a pose de pouso e o caminho de
   volta à base — e quedas de verdade gravadas da ginga treinada (falls.py)
   │  env.py + train_mjx.py   imitação de movimento com PPO (DeepMimic), no
   │                      Modal por padrão: a política vê o andamento e dá
   │                      alvos para os motores PD; a física decide o resto
   ▼
política treinada
   │  choose.py     para cada golpe, andamento e momento do pedido, testa em
   │                simulação cada perna e cada entrada e guarda a melhor
   │  check.py      minutos seguidos trocando de toque: quedas, deriva, pés,
   │                golpes; com --recover e --push-every, quedas e levantadas
   │  export_web.py modelo + pesos da rede + referências → public/data/sim/ginga/
   ▼
navegador (src/sim/simulation.ts): MuJoCo em WebAssembly + a rede, ao vivo
```

- **Modelo (model.py).** O mesmo esqueleto do app (mesmos ossos, mesmas
  medidas, mundo com Y para cima), 70 kg divididos pelos segmentos segundo de
  Leva (1996), juntas com os mesmos ângulos anatômicos e limites, pés com sola
  plana e atrito, motores de posição com torque máximo por articulação. O
  corpo é sólido: as partes colidem entre si (menos as vizinhas de uma mesma
  articulação), então o braço não atravessa a cabeça nem as pernas se cruzam
  uma por dentro da outra.
- **Andamento e estilo.** Os toques e a ginga mudam juntos (Capoeira Angola
  é lenta, baixa e fluida, perto do chão; a Regional, no São Bento Grande, é
  rápida, de pé, cadenciada — ver as fontes no README principal). A
  referência é gerada em cinco andamentos de 60 a 130 bpm, sempre um ciclo
  completo (os dois lados) por compasso de quatro tempos: em 60 bpm o quadril
  fica ~9 cm mais baixo que no vídeo, o tronco 15 % mais inclinado e o pé que
  recua sobe como no vídeo; em 130 bpm o quadril fica ~3 cm mais alto, o
  tronco 15 % menos inclinado e o passo sobe só 55 % — passos curtos e
  rápidos. Entre dois andamentos a referência é interpolada.
- **Triângulo.** Os cantos vêm do `triangle` em
  `public/data/movements/ginga.yaml` (o mesmo que o app desenha no chão): cada
  pé descansa no seu canto da frente e vai para trás no canto de trás, apoiado
  na ponta. O `reference.py` reajusta as pernas e a raiz da referência com
  cinemática inversa para isso; tronco, braços e tempo continuam os do vídeo.
- **Equilíbrio da referência.** Uma pose copiada do vídeo não precisa ser
  possível para um corpo com massa: na primeira versão o centro de massa
  ficava ~20 cm fora do pé de apoio durante metade do ciclo (o pé que recua
  está no ar), e nenhum controle segura isso — por isso a ginga lenta caía
  mais (fica mais tempo num pé só). Agora o `reference.py` calcula o caminho
  do centro de massa pelo modelo do pêndulo invertido: o ponto de onde o chão
  empurra (ponto de momento zero, c − (h/g)·c'') tem de ficar sob o pé de
  apoio enquanto o outro está no ar, e entre os pés no resto. Num ciclo que
  se repete isso é um filtro: cada harmônico do caminho dos apoios dividido
  por 1 + (h/g)·ω² — quanto mais rápido o toque, menos o corpo balança. A IK
  coloca o centro de massa nesse caminho; o ponto de momento zero ficou a
  2–7 cm do apoio (mediana; antes 20–56 cm). Os golpes fazem o mesmo, com
  0,3 s parados na base no começo e no fim para o peso passar para o pé de
  apoio enquanto os dois pés ainda estão no chão.

- **Golpes, esquiva, cair e levantar e perna e momento.** Em [README-MOVES.md](README-MOVES.md).
- **Recompensa, treino, navegador e resultado do treino.** Em [README-TRAINING.md](README-TRAINING.md).

## Rodando

```bash
uv sync
uv run python capture.py <vídeo> 4.5 19 data/ginga.npz
uv run python retarget.py data/ginga.npz 4.84 11.2 data/ginga_fit.npz       # câmera da frente
uv run python retarget.py data/ginga.npz 11.3 18.1 data/ginga_fit_side.npz  # câmera do lado
uv run python reference.py                                                  # → data/ginga_ref.npz
# em teste, no lugar do capture.py: GVHMR na GPU do Modal, uma câmera, com o caminho do corpo no mundo
uv run modal run hmr_modal.py --video data/youtube/ZCLVQIfaPHE.mp4 --start 7.3 --end 12.2 --out data/martelo_hmr.npz --static
# vários vídeos de uma vez: cortes de cena → GVHMR em paralelo → que golpes cada trecho tem e onde começam e acabam de pé
uv run python video_shots.py --intro 6.8 data/youtube/*.mp4 > data/youtube/segments.json
uv run modal run hmr_modal.py::batch --jobs data/youtube/segments.json --out data/hmr
uv run python clip_score.py data/hmr/*.npz --json data/hmr/moves.json
uv run python clip_speed.py data/hmr/ZCLVQIfaPHE_6.8s.npz 4 13.0 41.0 data/martelo_real.npz    # o replay em câmera lenta no tempo real
uv run python retarget.py data/martelo_real.npz 13.0 19.97 data/martelo_fit_real.npz           # martelo, Capoeira Nantes
uv run python plant_clip.py data/martelo_real.npz data/martelo_fit_real.npz 13.7 15.1 data/martelo_r_ref_real.npz  # do golpe ao pouso, pés plantados, peso sobre o apoio
uv run python clip_mirror.py data/martelo_real.npz data/martelo_real_mirror.npz             # a outra perna, espelhada (o vídeo só mostra uma bem)
uv run python retarget.py data/martelo_real_mirror.npz 13.0 19.97 data/martelo_fit_real_mirror.npz
uv run python plant_clip.py data/martelo_real_mirror.npz data/martelo_fit_real_mirror.npz 13.7 15.1 data/martelo_l_ref_mirror.npz
# no Linux sem tela, o viz.py precisa de MUJOCO_GL=egl
uv run modal run --detach modal_train.py --no-wait --skill ginga --minutes 180 --out runs/ginga_tempo --init runs/ginga_tri/last.pt --log-std -2.0
uv run modal run modal_train.py::fetch --run ginga_tempo                    # traz best.pt, last.pt e log.jsonl para runs/
(cd .. && npx --yes tsx@4.19.2 scripts/sample-clips.ts martelo bencao > ml/data/clips.json)
uv run python kicks.py                                                      # → data/kicks.npz
uv run modal run --detach modal_train.py --no-wait --skill all --minutes 240 --out runs/ginga_moves --init runs/ginga_tempo/last.pt --log-std -2.0
uv run modal run modal_train.py::fetch --run ginga_moves
# sem o Modal, no próprio Mac: uv run python train_mjx.py ... (mesmas opções) ou train.py (PPO na CPU)
uv run python check.py runs/ginga_moves/last.pt                              # quedas, deriva, por toque
uv run python choose.py runs/ginga_moves/last.pt                             # → data/choices.json
uv run python export_web.py runs/ginga_moves/last.pt ../public/data/sim/ginga  # → navegador
uv run python viz.py data/ginga_ref.npz ref.png                             # conferir poses
uv run python check.py runs/ginga_moves/last.pt --minutes 0.5 --record run.npz && uv run python viz.py run.npz run.png 6 7 8
                                     # o corpo treinado (em cima) e a referência que ele seguia (embaixo)
```

Para dividir o trabalho entre agentes do Claude Code, `.claude/agents/` tem o
**trainer** (treina, testa, exporta; não edita código) e o **reviewer** (julga
o movimento em imagens e no navegador; só lê), e `.agents/skills/` tem os
manuais que eles seguem (`capoeira-eval`, `capoeira-render`). A sessão
principal coordena e faz as mudanças de código. Inicie o Claude na raiz do repositório
para que eles apareçam.

O vídeo, os pontos extraídos dele, os modelos baixados e os checkpoints não
vão para o git (`.gitignore`); só o código e a simulação final exportada.
