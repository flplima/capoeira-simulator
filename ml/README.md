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
   │  env.py + train.py   imitação de movimento com PPO (DeepMimic): a política
   │                      vê o andamento e dá alvos para os motores PD; a
   │                      física decide o resto
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
- **Golpes (kicks.py, env.py).** O martelo e a bênção partem dos
  quadros-chave do app, que foram ajustados olhando as duas câmeras do vídeo
  nos pontos-chave de cada golpe, e conferidos com descrições técnicas: no
  martelo o joelho sobe como na bênção, o pé de apoio gira 90–180° na ponta,
  o quadril fica alinhado com a perna, o corpo inclina um pouco para trás e a
  canela estala no fim, acertando com o peito do pé na altura do ombro/cabeça
  ([Lalaue — martelo](https://lalaue.com/moves/martelo/)); a bênção é um
  empurrão frontal com a sola ou o calcanhar, na altura do peito, saindo da
  base ([Lalaue — bênção](https://lalaue.com/moves/bencao)). Ajuste feito por
  isso: no golpe do martelo o giro vai a 90° e a perna sobe mais (o peito do
  pé chega a ~1,2 m). Depois, comparados com fotos dos dois golpes
  ([martelo, Abadá](https://commons.wikimedia.org/wiki/File:CapoeiraMartelo_ST_05.jpg),
  [martelo num batizado](https://commons.wikimedia.org/wiki/File:Batizado_Planeta_Capoeira_Martelo.jpg),
  [bênção, Abadá](https://commons.wikimedia.org/wiki/File:CapoeiraBencao_ST_05.jpg)):
  o martelo girava o pé de apoio 75° com ele chapado no chão — o corpo
  simulado só gira um pé chapado arrastando os quatro cantos, e por isso não
  girava (o quadril virava 1–16° em vez de 77°, e a perna saía para o lado);
  agora o calcanhar de apoio sobe ~7 cm e o giro é na ponta do pé, o tronco
  inclina 27° para longe da perna (antes 18°) e o peito do pé chega a 1,37 m.
  A bênção inclinava o tronco 5° para a frente; agora inclina 15° para trás,
  com o pé de apoio chapado, e a sola chega à altura do peito (ponta do pé a
  1,36 m, antes 1,13 m). Cada golpe é amostrado a 30 quadros/s para as duas
  pernas (`scripts/sample-clips.ts`), convertido em ângulos do modelo e
  plantado por IK: o pé de apoio fica no seu canto da frente o tempo todo
  (girando no lugar), o pé que chuta sai do canto de trás e volta para ele.
  A ginga entra no golpe a partir da base com a perna que chuta atrás (fase 0
  para a direita, 0,5 para a esquerda, com até ±0,1 de folga), congela a fase
  enquanto o golpe acontece, mistura a pose da ginga com a do golpe nos
  primeiros e nos últimos 0,35 s e segue a ginga dali. O golpe se ancora no
  pé em que o corpo está de fato apoiado (até 25 cm do canto): a ginga nunca
  pisa exatamente no canto, e um golpe equilibrado sobre o canto com o pé a
  10 cm dali derrubava o corpo. Quem espera o momento é a rede da ginga; a
  rede dos golpes assume quando o golpe começa e devolve 2,5 s depois (como a
  da esquiva, abaixo; devolvendo em 1 s, ~60 % das quedas vinham 1,4–2,5 s
  depois do golpe, já na rede da ginga), e quando o golpe acaba o triângulo
  vai para debaixo dos pés. O treino dos golpes começa no golpe: metade dos
  episódios no primeiro quadro (com a pose um pouco perturbada e o pé um
  pouco fora do canto, como na passagem da ginga), metade no meio dele —
  ou, em 30 %, na ginga esperando o momento — e cada um segue 3 s de ginga
  depois. Os tamanhos sobem (40 → 60 → 80 → 100 % do golpe) quando 75 % dos
  golpes inteiros terminam de pé. A política recebe qual golpe, qual perna, o
  quanto do golpe já passou e a mistura.
- **Esquiva lateral (esquiva.py, env.py).** Também sai da base e volta para
  ela, como um golpe, e tem rede própria (`--skill esquiva`): da base com a
  perna direita atrás, o corpo desce de uma vez sobre a perna da frente
  dobrada enquanto a de trás abre esticada para o lado num passo baixo
  (2–2,5 cm do chão), o tronco deita sobre o joelho com a cabeça para fora,
  para a esquerda, os olhos no camarada e o antebraço direito sobre a
  cabeça; segura, sobe na perna da frente e o pé de trás volta para o canto.
  Desce tão rápido quanto a gravidade e as pernas deixam (o quadril cai com
  no máximo 7 m/s² e freia com no máximo 9 m/s²): a cabeça passa de 1 m em
  ~0,33 s (um golpe chega em 0,3–0,4 s) e o fundo vem em ~0,5 s, quadril a
  ~0,5 m e cabeça a ~0,76 m, pés a ~1 m um do outro — nos cantos da frente do
  triângulo (0,68 m) uma perna esticada seguraria o quadril a 0,8 m. Sobe
  mais devagar (no máximo 4 m/s² para cima e 6 m/s² para frear: subir de uma
  perna só bem dobrada é o difícil, e um corpo que não acompanha a subida
  senta), e o pé de trás só volta para o canto com o quadril quase em cima;
  1,8 s ao todo. Parte dos quadros-chave do app (`esquiva-lateral.yaml`), que a IK
  planta: o pé da frente fica no seu canto, o de trás sai do canto num passo
  sem tranco e pousa chapado; o centro de massa recebe um empurrão curto do
  pé de trás e rola como um pêndulo até o pé da frente enquanto o de trás
  está no ar (o ponto de momento zero fica no pé da frente); altura do
  quadril e da cabeça, a inclinação do tronco, solas chapadas e pernas e
  mãos longe do chão também entram no ajuste. Tamanhos para o currículo:
  40–100 % da esquiva. A política recebe as mesmas entradas de um golpe e
  mais uma, no fim, que diz que é a esquiva. A rede da esquiva continua
  gingando 2,5 s depois dela antes de devolver o corpo à rede da ginga, e treina 3 s de ginga depois de cada esquiva: devolvido 1 s
  depois, o corpo ainda não estava numa ginga que a rede da ginga
  segurasse, e caía 1,5–2,7 s depois da esquiva. Treinada só a partir da
  esquiva, a rede esquecia a ginga (sozinha, caía 4–12 vezes em 30 s a cada
  toque; a da ginga, nenhuma): por isso 30 % dos episódios começam na ginga
  e esperam o momento de entrar, e o treino parte da rede da ginga com
  taxa de aprendizado menor (1e-4). E quando a esquiva acaba o
  triângulo da ginga vai para debaixo dos pés, onde eles de fato estão (como
  depois de levantar do chão; sem girar — girar pela linha entre os pés
  virava a ginga 8–20° a cada esquiva, e o corpo saía andando atrás): o passo largo nunca
  volta exato para o canto, e uma ginga que continua pisando num triângulo
  a 0,1–0,4 m dali sai andando e cai. Fora de uma queda, quadril
  abaixo de 0,4 m ou cabeça abaixo de 0,8 m é queda — ou, onde a referência
  desce mais que isso, 0,3 m abaixo dela.
- **Cair e levantar (floor.py, falls.py, env.py).** Capoeirista que perde o
  equilíbrio não desaba: caindo para trás, senta na queda de quatro (mãos e
  pés no chão, barriga para cima); para o lado, desce na negativa (uma perna
  dobrada por baixo, a outra esticada, a mão do lado no chão); para a frente,
  apara a queda nas mãos e cai de joelhos. As poses de pouso das duas
  primeiras vêm dos quadros-chave do app; a terceira e os três caminhos de
  volta têm os quadros-chave no próprio `floor.py`, feitos para que o corpo
  nunca precise se jogar do chão para cima: mão ou pé só sai do chão com o
  peso já sobre o que fica. Da queda de quatro, as mãos empurram o quadril
  para a frente, por cima dos pés, até a cocorinha, e o corpo sobe dali; da
  negativa, a perna esticada se recolhe arrastando o calcanhar e planta o pé
  à frente, a mão empurra o peso para ele e o corpo sobe de uma base baixa
  com a perna dobrada atrás; de quatro apoios, um pé vem à frente ao lado da
  mão, o quadril passa por cima dele, as mãos empurram o chão numa base baixa
  e o corpo sobe com o outro pé atrás. A IK põe pouso e caminho no chão de
  trás para a frente: o último quadro é a base da ginga no
  triângulo, e em cada quadro anterior as mãos, joelhos e pés que sustentam o
  peso tocam o chão sem escorregar (só o calcanhar arrastado desliza) — um apoio
  que sai do chão fica onde tocou no pouso, e o peso passa para os outros aos
  poucos —, com o centro de massa sobre eles, e já sobre os que ficam antes
  que um deles saia. Onde os quadros-chave sobem rápido demais para um corpo de
  torque limitado, o clipe se alonga: o centro de massa nunca sobe a mais de
  0,8 m/s nem anda a mais de 1 m/s (o caminho de volta dura 3,5–6,2 s). O
  `falls.py` grava 3000 perdas de equilíbrio de verdade da ginga e dos golpes
  treinados, com empurrões de 120–450 N de um lado qualquer. Perder o
  equilíbrio é algo além de mãos e pés no chão, a cabeça baixa, ou o quadril
  inclinado 0,65 rad além do da referência (só a inclinação: o giro do
  martelo sobre o pé de apoio não é queda), 25 cm afundado ou 0,5 m longe
  dela. Para onde o corpo vai é para onde o centro de massa passa do meio dos
  pés, mais a velocidade do quadril, visto de onde o corpo está virado (não a
  inclinação do quadril, que na ginga já pende para a frente): para a frente,
  mãos e joelhos; para trás, queda de quatro; para o lado, negativa, dobrando
  a perna que carrega o peso (o pé mais perto de para onde o corpo tomba) e
  esticando a livre — dobrar a livre estica a de apoio, e o corpo só pode
  girar por cima dela —, mas um tronco já dobrado mais de 0,9 rad para a frente (do quadril
  à cabeça) vai para mãos e joelhos de qualquer jeito: sentar numa negativa
  com a cabeça já sobre as mãos é mergulhar de cabeça. A referência vai da
  pose do momento até a pose de pouso em 0,5 s (rápido no começo e
  assentando no pouso, mas nunca à frente da gravidade: o quadril não desce
  mais do que desceria em queda livre a partir da velocidade que tinha, e
  a pose sobe o que for preciso para nenhum pé, joelho ou mão passar do chão
  no caminho; os
  braços vão logo buscar o chão: caindo para trás, as mãos estão atrás do
  quadril e perto do chão em 0,3 s),
  virada para onde o corpo está virado e um pouco adiante na direção da queda;
  ao pousar, o caminho de volta começa onde o corpo de fato está, sobe e
  entrega a ginga, que continua onde o corpo levantou (o triângulo vai junto).
  No treino, só a cabeça e o pescoço não podem tocar o chão (o episódio
  acaba se, depois de pousar, o corpo se afasta 0,6 m do caminho de volta, se
  fica 1 s deitado enquanto o caminho de volta espera por ele, ou se, descendo ou
  logo depois de pousar, a cabeça fica 30 cm abaixo da da referência — o
  corpo que segura o quadril no alto e tomba por cima dele mergulha de
  cabeça e só bate um instante depois), e durante a queda
  parte da nota vem de descer devagar (quadril, peito e cabeça sem despencar)
  e com a cabeça acima do quadril (o quadril desce primeiro; mergulhar de
  cabeça é como ela bate no chão). As quedas gravadas começam devagar: de
  início com o corpo parado na pose em que perdeu o equilíbrio, depois com
  35 %, 70 % e toda a velocidade, a cada vez que metade delas termina de pé.
  Os episódios começam numa queda gravada (50 %), na pose de pouso (20 %) ou em
  qualquer ponto do caminho de volta (30 %, em qualquer lugar e direção do
  chão) e seguem 2 s de ginga depois de levantar. A rede de recuperação é a
  terceira rede: assume ao perder o equilíbrio e devolve 1 s depois de
  levantar.
- **Perna e momento (choose.py).** Depois do treino, para cada golpe, cinco
  andamentos e 16 momentos do pedido dentro do ciclo, a política tenta, em
  simulação, as duas pernas e cinco momentos de entrada a partir de uma ginga
  já estabilizada. Fica a opção que melhor equilibra ficar de pé (até 1 s
  depois do golpe), acompanhar bem o golpe e responder rápido (perde 0,15 de
  nota por segundo de espera). O app só consulta essa tabela.
- **Recompensa (env.py).** Quão perto a simulação fica da referência: ângulos
  (40 %), velocidades (10 %), posição de mãos, pés e cabeça (15 %), da raiz
  (15 %) e de cada pé no seu ponto do triângulo, no mundo (20 %). As mãos,
  pés e cabeça são medidos virados para onde o corpo olha, então durante um
  golpe 20 % da recompensa vai para onde o pé do chute está em relação ao
  quadril sem descontar esse giro (`strike`): um martelo que sai para o lado
  sem o quadril virar perde ali o que ganhava nos outros termos. O episódio
  acaba quando algo além de mãos e pés toca o chão, a cabeça desce demais ou o
  corpo se afasta 0,5 m da referência; começa em uma fase e um andamento
  aleatórios e dura até 40 s, com o toque mudando de vez em quando (em média
  a cada 8 s), para a política aprender a acompanhar a música quando ela
  muda. Como o triângulo é fixo no mundo, ela aprende a gingar no lugar.
- **Treino (train.py).** PPO com GAE, rede MLP de 512–256 para política e
  valor, observações normalizadas; 12 simulações em paralelo na CPU.
  `--init` parte de um checkpoint anterior (entradas novas começam com peso
  zero).
- **No navegador (src/sim/simulation.ts).** O mesmo `humanoid.xml` roda no
  MuJoCo 3.15 compilado para WebAssembly (`@mujoco/mujoco`, a mesma versão do
  Python). A cada 0,03 s o app monta a observação exatamente como o `env.py`,
  passa pela rede (três camadas, 204 mil pesos, em JavaScript puro) e aplica
  os alvos aos motores; 15 passos de física de 2 ms depois, repete. A fase da
  ginga é puxada suavemente para o compasso do berimbau quando a música toca.
  Se o corpo cair, ele é posto de pé de novo depois de 1,2 s. O teste
  `test/live.test.ts` confere observação, ação e estado seguinte contra
  valores calculados em Python.

## Resultado do treino

Primeira rodada (sem triângulo): 110 minutos, 22 milhões de passos; os
episódios foram de 0,8 s a ~17 s e a recompensa de 0,30 a ~0,55, mas o corpo
escorregava até 1,7 m de lado em quatro ciclos.

Segunda rodada (triângulo), partindo da primeira: 120 minutos, 20 milhões de
passos. A recompensa foi de 0,38 a 0,87 (pés no triângulo 0,33 → 0,88). A
política final gingou 60 ciclos seguidos (216 s) sem cair, sem se afastar
mais de 0,2 m do centro do triângulo, com o pé apoiado a ~3 cm do canto
(mediana). O movimento entra num ciclo estável: o estado no fim do trecho
exportado é praticamente igual ao do início, então o loop no app não tem
emenda — é uma ginga sem fim.

Cair e levantar (rede de recuperação), partindo da ginga: três testes de 15
minutos e uma rodada de 4 horas (100 milhões de passos). No treino, a parte
dos episódios que termina de pé subiu de 0,26 para ~0,36 — mas lá muitos
começam no meio do caminho de volta. No teste de 3 minutos (ginga e golpes
das melhores redes, empurrões de 350 N a cada 4 s, golpes a cada 5 s, seis
andamentos de 66 a 120 bpm) o corpo levantou de 0 de 29 quedas (a rodada de
4 h; a anterior, 1 de 30): em ~70 % delas a cabeça toca o chão logo depois
do pouso (0,4–0,9 s após perder o equilíbrio) e o resto se perde no chão. O
corpo não chega à pose de pouso da referência — rola de costas de quatro
apoios, cai de cabeça na negativa. A meta é 80 % de pé com no máximo 10 % de
cabeça no chão; até lá o app põe o corpo de pé de novo. Com empurrões, os
golpes terminam de pé martelo 10 de 13 e bênção 9 de 13 com a rede de
recuperação, e 11 de 18 e 15 de 18 sem ela — diferença dentro do ruído de
amostras tão pequenas. Sem empurrões, a
rede de golpes termina de pé 16 de 18 martelos e 9 de 18 bênçãos; empurrado,
o corpo se afasta 0,5–0,9 m do centro do triângulo.

Esquiva lateral (rede própria), partindo da ginga: cinco testes de 15 minutos
e uma rodada de 4 horas (40 milhões de passos, taxa 1e-4). O tamanho chegou
à esquiva inteira em 20 minutos; a recompensa foi de 0,64 a 0,88 e a parte
dos episódios que chega ao fim do relógio de 0,52 a 0,94. No teste de 5
minutos (a ginga da melhor rede, esquiva inteira pedida a cada 5 s, cinco
andamentos de 60 a 130 bpm) terminou de pé 58 de 60 esquivas, com uma queda
só (a 60 bpm). A 66/90/120 bpm, com a esquiva a cada 5 s por 1 minuto em
cada um, 36 de 36 sem queda; a rodada anterior, treinada só até 60 % da
esquiva, caía 33 vezes no tamanho inteiro. Sozinha, a rede da esquiva ginga
como a da ginga (0 quedas de 66 a 130 bpm, pé a 2,4–3,0 cm do canto). Nas
imagens o corpo segue a referência com ~2° de erro nas juntas, desce ~0,06 s
atrás dela, o pé de trás dá o passo (2,5–7,6 cm do chão) em vez de arrastar,
a mão livre roça o chão ~0,1 s no fundo e a ginga depois volta limpa (raiz a
0,04–0,08 m da referência).

Golpes de novo (martelo e bênção), com a mesma receita da esquiva: a rede
dos golpes ginga 2,5 s depois do golpe, treina 3 s disso, 30 % dos episódios
começam na ginga e o triângulo vai para debaixo dos pés no fim do golpe. A
rede anterior não sabia gingar (sozinha, 35 quedas em 3 minutos, o pé a
~14 cm do canto), e as quedas vinham 1,4–2,5 s depois do golpe. Partindo da
ginga, 15 minutos não saíram do menor tamanho; partindo da rede anterior,
um teste de 15 minutos e uma rodada de 4 horas (taxa 1e-4): o tamanho chegou
ao golpe inteiro em 30 minutos e a recompensa foi de 0,52 a 0,74 (ainda
subindo no fim). No teste de 5 minutos (golpe inteiro a cada 5 s, cinco
andamentos de 60 a 130 bpm) termina de pé martelo 23 de 28 e bênção 24 de
27, com 13 quedas (a rede anterior, no mesmo código: 11 de 30, 21 de 29 e 38
quedas). Sozinha, a nova rede ginga sem cair, mas com o pé a 5–13 cm do
canto. Nas imagens o martelo e a bênção leem como os de verdade (~3° de
erro nas juntas) e voltam para uma ginga limpa; a bênção da perna direita
sobe ~15 cm a menos que a referência, e o pé de apoio gira menos que o da
referência no martelo e escorrega até 0,2 m.

Golpes com a postura das fotos (ver **Golpes** acima: calcanhar de apoio no
alto no giro do martelo, tronco mais inclinado, bênção inclinada para trás, e
os termos `strike` e `pivot` na recompensa). Medida no pico de cada golpe, a
rede anterior não virava o quadril no martelo (1–16° contra 77° da
referência; o pé de apoio girava 6–10°, às vezes para o lado errado) e a
perna saía para o lado (65–89° da frente). Dois testes de 15 minutos (o
segundo já com `pivot`, que tirou o giro do pé do lado errado) e uma rodada
de 4 horas (taxa 1e-4; a recompensa foi de 0,63 a 0,79, `strike` 0,84 →
0,91, `pivot` 0,82 → 0,92). Com o `best.pt` o martelo vira: quadril 64° e
−50° (referência ±72°), pé de apoio 38° e −28° (27°), perna a 19° e 40° da
frente (18°), a 1,25–1,43 m; a bênção fica na referência (1,31–1,37 m,
tronco 15–23° para trás). Na CPU o equilíbrio ficava na beira da barra (martelo
11 de 16 e bênção 14 de 16 no `best.pt`, ou 15 de 17 e 10 de 17 no `last.pt`);
continuando no MJX em GPU L4 (512 ambientes, ~7 900 passos/s) até a amplitude
inteira (`mjx_kicks5`, 35 min a 1e-4) seguido de um polimento curto com menos
ruído (`mjx_kicks6`, 20 min a 3e-5, `log_std` −2,6), o `last.pt` termina de pé
martelo 15 de 16 (1 queda) e bênção 14 de 16 (1 queda) no teste de 3 minutos
(66, 90 e 120 bpm; barra: 14 de 16), e entrou no app.

## Rodando

```bash
uv sync
uv run python capture.py <vídeo> 4.5 19 data/ginga.npz
uv run python retarget.py data/ginga.npz 4.84 11.2 data/ginga_fit.npz       # câmera da frente
uv run python retarget.py data/ginga.npz 11.3 18.1 data/ginga_fit_side.npz  # câmera do lado
uv run python reference.py                                                  # → data/ginga_ref.npz
uv run python train.py --minutes 180 --out runs/ginga_tempo --init runs/ginga_tri/last.pt --log-std -2.0
(cd .. && npx --yes tsx@4.19.2 scripts/sample-clips.ts martelo bencao > ml/data/clips.json)
uv run python kicks.py                                                      # → data/kicks.npz
uv run python train.py --minutes 240 --out runs/ginga_moves --init runs/ginga_tempo/last.pt --log-std -2.0
uv run python check.py runs/ginga_moves/last.pt                              # quedas, deriva, por toque
uv run python choose.py runs/ginga_moves/last.pt                             # → data/choices.json
uv run python export_web.py runs/ginga_moves/last.pt ../public/data/sim/ginga  # → navegador
uv run python viz.py data/ginga_ref.npz ref.png                             # conferir poses
uv run python check.py runs/ginga_moves/last.pt --minutes 0.5 --record run.npz && uv run python viz.py run.npz run.png 6 7 8
                                     # o corpo treinado (em cima) e a referência que ele seguia (embaixo)
```

Para dividir o trabalho entre agentes do Claude Code, `.claude/agents/` tem o
**trainer** (treina, testa, exporta; não edita código) e o **reviewer** (julga
o movimento em imagens e no navegador; só lê), e `.claude/skills/` tem os
manuais que eles seguem (`capoeira-eval`, `capoeira-render`). A sessão
principal coordena e faz as mudanças de código. Inicie o Claude em `capoeira/`
para que eles apareçam.

O vídeo, os pontos extraídos dele, os modelos baixados e os checkpoints não
vão para o git (`.gitignore`); só o código e a simulação final exportada.
